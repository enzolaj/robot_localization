#!/usr/bin/env python3

""" This is the starter code for the robot localization project """
from matplotlib import pyplot as plt
import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler
from scipy.ndimage import gaussian_filter		# blurs map walls into likelihood field

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """Constructs a new Particle instance.

        Args:
            x (float): x-coord of the hypothesis in map frame [m] (default = 0.0).
            y (float): y-coord of the hypothesis in map frame [m] (default = 0.0).
            theta (float): Yaw of the hypothesis in map frame [rad] (default = 0.0).
            w (float): Particle weight, unnormalized [unitless] (default = 1.0).
        """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """Converts the particle state into a ROS geometry_msgs/Pose message.

        Returns:
            Pose: ROS Pose message with the particle's 2D position (z = 0.0) and quaternion orientation.
        """
        q = quaternion_from_euler(0, 0, self.theta)		# roll = 0, pitch = 0, yaw = theta
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

    # particle_cloud is kept as one N x 3 numpy array so math is fast.
    # this class is just used to wrap a single row into a Pose msg when publishing to rviz

class ParticleFilter(Node):
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            last_scan_timestamp: this is used to keep track of the clock when using bags
            scan_to_process: the scan that our run_loop should process next
            occupancy_field: this helper class allows you to query the map for distance to closest obstacle
            transform_helper: this helps with various transform operations (abstracting away the tf2 module)
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            thread: this thread runs your main loop

            particle_cloud is stored as an N x 3 numpy array, one row [x (m), y (m), theta (rad)] per particle, in the map frame
            particle_weights: one weight per particle, in the same order as particle_cloud, adding up to 1
            likelihood_field: a blurred copy of the map, indexed [x cell, y cell], that is high on and near walls
            laser_offset: the (x, y) position of the laser on the robot, relative to its center
    """
    def __init__(self):
        """Initializes the ParticleFilter ROS node, parameters, subscribers, and worker thread."""
        super().__init__('pf')
        self.base_frame = "base_footprint"		# robot base frame
        self.map_frame = "map"		# global map coordinate frame
        self.odom_frame = "odom"		# odometry coordinate frame
        self.scan_topic = "scan"		# lidar scan topic

        # initialize particle field
        self.n_particles = self.declare_parameter("n_particles", 6000).value		# total number of particles

        # motion thresholds before triggering a filter update
        self.d_thresh = self.declare_parameter("d_thresh", 0.05).value		# linear movement threshold [m]
        self.a_thresh = self.declare_parameter("a_thresh", math.pi/24).value		# angular movement threshold [rad]

        # resampling params, keep a fraction random to prevent particle degeneracy
        # and use smaller spread than init since resampled particles already scored well
        self.random_resample = self.declare_parameter("random_resample", 0.2).value		# ratio of random particles [0.0 to 1.0]
        self.resample_xy_sigma = self.declare_parameter("resample_xy_sigma", 0.1).value		# resample x/y std dev [m]
        self.resample_theta_sigma = self.declare_parameter("resample_theta_sigma", 0.1).value		# resample yaw std dev [rad]

        # scoring laser scans
        self.sigma_hit = self.declare_parameter("sigma_hit", 0.3).value		# gaussian blur std dev from walls [m]
        self.min_likelihood = self.declare_parameter("min_likelihood", 0.01).value		# floor prob [0.0 to 1.0] so one bad reading cant hurt a good particle
        self.log_scale = self.declare_parameter("log_scale", 20.0).value		# divides summed log scores since nearby beams hit the same wall and arent independent
        self.beam_step = self.declare_parameter("beam_step", 3).value		# take every k-th beam for computational speed

        # starting cloud spread around a 2d pose estimate click in rviz if supplied
        self.init_xy_sigma = self.declare_parameter("init_xy_sigma", 0.3).value		# initial x/y std dev [m]
        self.init_theta_sigma = self.declare_parameter("init_theta_sigma", 0.3).value		# initial yaw std dev [rad]

        # odometry noise as a fraction of distance driven or angle turned since last update
        self.odom_trans_noise = self.declare_parameter("odom_trans_noise", 0.1).value		# translational noise [m/m]
        self.odom_rot_noise = self.declare_parameter("odom_rot_noise", 0.1).value		# rotational noise [rad/rad]

        # pose estimate clustering
        self.cluster_radius = self.declare_parameter("cluster_radius", 1.0).value		# radius around best particle to use for averaging [m]

        # pose_listener responds to 2D Pose Estimate clicks in rviz if using initial pose
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # publish current particle cloud so we can view particles in rviz
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # keep track of timestamps from bag files so map -> odom transform uses bag time instead of wall time
        self.last_scan_timestamp = None
        self.scan_to_process = None     # current LaserScan msg for run_loop to process
        self.particle_cloud = []        # N x 3 array, one row [x (m), y (m), theta (rad)] per particle
        self.particle_weights = []      # length N array of weights [unitless], sums to 1.0

        self.current_odom_xy_theta = []		# [x (m), y (m), theta (rad)] in odom frame
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)

        # build likelihood field once so scoring is just lookup
        self.build_likelihood_field()

        # use a separate thread to work around single-threaded execution bottleneck in ROS2
        # daemon=True lets program exit on ctrl+c since loop_wrapper runs an infinite while loop
        thread = Thread(target=self.loop_wrapper, daemon=True)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)		# 0.05 s period = 20 Hz

        # old testing/visualization code
        #self.initialize_particle_cloud()
        # self.particle_weights = np.zeros([self.n_particles, 1])
        # self.particle_weights[10:13, 0] = 0.1
        # self.particle_weights[0, 0] = 0.7
        # self.resample_particles()

    def pub_latest_transform(self):
        """Broadcasts the latest map to odom transform with a 0.1 s future timestamp.

        Postdating the transform by 0.1 s prevents TF extrapolation errors in RViz
        while waiting for the next laser scan update.
        """
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)

    def loop_wrapper(self):
        """Calls run_loop repeatedly at ~10 Hz on the background thread.

        Using a separate thread works around single-threaded bottlenecks in ROS2
        so slower filter updates do not block incoming sensor callbacks.
        """
        while True:
            self.run_loop()
            time.sleep(0.01)		# [s]

    def run_loop(self):
        """Main processing loop of the particle filter.

        Checks if scan_to_process is available, matches it to an odometry pose via TF,
        subsamples valid laser beams in [range_min, range_max), looks up the laser's
        (x, y) mounting offset, and runs the odom, laser, pose, and resample steps
        once movement exceeds d_thresh or a_thresh. The particle cloud is only published
        after it is first made or updated, since it does not change on other scans.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame, self.base_frame, msg.header.stamp)
        if new_pose is None:
            # unable to get robot odom pose at the exact scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):		# delta_t < 0.0 s means scan is older than tf buffer
                # we will never get this transform since it is before our oldest one, so drop the scan
                self.scan_to_process = None
            return

        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)		# r [m], theta [rad]
        self.get_logger().debug(f"r[0]={r[0]}, theta[0]={theta[0]}")

        # keep every beam_step-th beam and skip readings outside valid laser range [range_min, range_max) [m]
        good_r = []
        good_theta = []
        for k in range(0, len(r), self.beam_step):
            if msg.range_min <= r[k] < msg.range_max:
                good_r.append(r[k])
                good_theta.append(theta[k])
        r = np.array(good_r)		# valid beam distances
        theta = np.array(good_theta)		# valid beam angles

        # lookup where the laser sits on the robot (x, y) [m]
        self.laser_offset = self.get_laser_offset(msg.header.frame_id)

        # clear current scan so scan_received can queue the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)		# [x (m), y (m), theta (rad)]
        self.get_logger().debug("odom pose x: {0:.2f}, y: {1:.2f}, yaw: {2:.2f}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif len(self.particle_cloud) == 0:
            # now that we have all necessary transforms we can initialize the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
            self.publish_particles(msg.header.stamp)		# show the starting cloud in rviz
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            # moved past d_thresh or a_thresh, run filter update
            # perf_counter is wall time, so it only lines up with lag (bag time) when the bag plays at --rate 1
            update_start = time.perf_counter()		# [s]
            self.update_particles_with_odom()		# move particles by odom delta
            self.update_particles_with_laser(r, theta)		# score particles against likelihood field
            self.update_robot_pose()		# estimate robot pose and update map -> odom
            self.resample_particles()		# resample cloud to focus on high-weight areas
            self.get_logger().debug(f"filter update took {time.perf_counter() - update_start:.3f} s")

            # only publish after an update since the cloud doesnt change otherwise, which saves ~0.06 s per skipped scan
            publish_start = time.perf_counter()		# [s]
            self.publish_particles(msg.header.stamp)
            self.get_logger().debug(f"publish took {time.perf_counter() - publish_start:.3f} s")
        lag = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e9		# ns to [s]
        self.get_logger().debug(f"lag {lag:.2f} s")

    # slightly modified from initial case for angle updates
    def moved_far_enough_to_update(self, new_odom_xy_theta):
        """Checks if the robot drove more than d_thresh or turned more than a_thresh.

        Args:
            new_odom_xy_theta (Sequence[float]): Current [x (m), y (m), theta (rad)] pose in odom frame.

        Returns:
            bool: True if linear change > d_thresh or angular change > a_thresh.
        """
        return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
               math.fabs(self.transform_helper.angle_diff(new_odom_xy_theta[2], self.current_odom_xy_theta[2])) > self.a_thresh


    def update_robot_pose(self):
        """Updates self.robot_pose and recalculates the map -> odom transform.

        The pose comes from estimate_pose, which finds the highest-weight particle and
        averages it with all particles within cluster_radius.
        """
        # make sure particle weights sum to 1.0 first
        self.normalize_particles()

        x, y, theta = self.estimate_pose()		# estimated pose in map frame
        self.robot_pose = Particle(x, y, theta).as_pose()
        if hasattr(self, 'odom_pose'):
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

    def update_particles_with_odom(self):
        """Updates all particles using the odometry change delta (dx, dy, dtheta).

        Since delta is measured in the odom frame, it is first rotated by the robot's
        old heading into the local body frame (forward, left, turn).
        That motion is then applied along each particle's own heading with Gaussian
        noise scaled by distance driven and angle turned, updating every particle at
        once as numpy arrays instead of looping over them.
        """
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # compute change in (x (m), y (m), theta (rad)) since last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            # just update pose to new pose if no current pose exists
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        # rotate delta by old_heading to get move in robot's local frame for both forward and left
        old_heading = old_odom_xy_theta[2]
        forward = math.cos(old_heading) * delta[0] + math.sin(old_heading) * delta[1]		# forward displacement [m]
        left = -math.sin(old_heading) * delta[0] + math.cos(old_heading) * delta[1]		# lateral left displacement [m]
        turn = self.transform_helper.angle_diff(new_odom_xy_theta[2], old_heading)		# yaw change [rad]

        # odom gets less reliable over longer moves, so scale noise std dev by distance and turn
        distance = math.hypot(forward, left)		# euclidean distance driven [m]
        trans_noise = self.odom_trans_noise * distance		# translational std dev [m]
        rot_noise = self.odom_rot_noise * abs(turn)		# rotational std dev [rad]

        n = len(self.particle_cloud)
        heading = self.particle_cloud[:, 2]		# every particle's heading, length N

        # give each particle delta in each direction plus additional scaled noise so that the cloud moves with more info
        # passing n to np.random.normal draws a different noise sample for every particle at once
        noisy_forward = forward + np.random.normal(0, trans_noise, n)		# length N [m]
        noisy_left = left + np.random.normal(0, trans_noise, n)		# length N [m]
        noisy_turn = turn + np.random.normal(0, rot_noise, n)		# length N [rad]

        # rotate local move by each particle's own heading to update its map (x, y)
        self.particle_cloud[:, 0] += noisy_forward * np.cos(heading) - noisy_left * np.sin(heading)
        self.particle_cloud[:, 1] += noisy_forward * np.sin(heading) + noisy_left * np.cos(heading)

        # wrap to [-pi, pi], uses np.arctan2 since angle_normalize uses math.atan2 which only takes one number
        new_heading = heading + noisy_turn
        self.particle_cloud[:, 2] = np.arctan2(np.sin(new_heading), np.cos(new_heading))

    def resample_particles(self):
        """Resamples the particle cloud according to particle_weights.

        Spawns random_resample fraction of particles uniformly across the map to
        prevent particle degeneracy, and fills the remaining normal_particle_count
        slots by drawing Gaussian copies (resample_xy_sigma, resample_theta_sigma)
        from the highest-weighted particles. Resets weights to 1 / n_particles afterward.
        """
        # make sure distribution sums to 1.0
        self.normalize_particles()

        # first determine the number of random particles and make that many at once
        random_particle_count = math.floor(self.n_particles * self.random_resample)
        random_particle_cloud = self.random_particles(random_particle_count)		# shape (N_rand, 3)

        # treat remaining particles as normal particles based on previous sample
        normal_particle_count = self.n_particles - random_particle_count

        # ravel flattens weights into 1D row so argsort works whether shape is (N,) or (N, 1)
        particle_weights_sort = np.ravel(self.particle_weights).argsort()
        sorted_weights = np.ravel(self.particle_weights)[particle_weights_sort[::-1]]		# descending order
        sorted_particles = self.particle_cloud[particle_weights_sort[::-1]]
        total_normal_weights = normal_particle_count / self.n_particles		# target weight sum [0.0 to 1.0]

        # find how many top weights it takes to reach total_normal_weights
        # cumsum[k] is the sum of the top k + 1 weights, and searchsorted finds the first spot it reaches the target
        # min() stops at the last particle in case float rounding keeps sum slightly under
        weight_sums = np.cumsum(sorted_weights)
        weights_to_use = min(np.searchsorted(weight_sums, total_normal_weights) + 1, len(sorted_weights))
        self.get_logger().debug(f"weights_to_use: {weights_to_use}")

        # kept weights only sum to ~total_normal_weights, so scale them to sum to 1.0
        # otherwise the copies below would only fill that fraction of normal_particle_count
        normal_weights = sorted_weights[:weights_to_use] / np.sum(sorted_weights[:weights_to_use])
        self.get_logger().debug(f"normal_weights: {normal_weights}")
        copies = (normal_weights * normal_particle_count).astype(int)		# copies of each kept particle

        # fix with LLM: int() rounds every count down, leaving a few slots unfilled
        # give leftover slots to the heaviest particles in order, 1 each. each count loses less than 1
        # when rounded down, so there are always fewer leftovers than kept particles
        copies[:normal_particle_count - np.sum(copies)] += 1

        # np.repeat lists kept particle j copies[j] times in a row, then each copy gets its own gaussian noise
        centers = np.repeat(sorted_particles[:weights_to_use], copies, axis=0)		# shape (N_norm, 3)
        normal_particle_cloud = self.normal_particles(centers, self.resample_xy_sigma, self.resample_theta_sigma)

        self.particle_cloud = np.concatenate((random_particle_cloud, normal_particle_cloud), axis=0)
        self.get_logger().debug(f"particle_cloud: {self.particle_cloud}")

        # every new particle starts with equal weight (1/N) until the next laser update scores them
        self.particle_weights = np.ones(self.n_particles) / self.n_particles

    def update_particles_with_laser(self, r, theta):
        """Updates particle_weights by scoring laser scan endpoints against likelihood_field.

        Casts the scan (r, theta) from every particle onto the map at once as an N x B array
        (N particles, B beams) and looks up every beam endpoint in likelihood_field. Sums
        log-probabilities across beams so multiplying hundreds of small probabilities does not
        round down to 0.

        Args:
            r (np.ndarray): Obstacle distance readings [m].
            theta (np.ndarray): Beam angles relative to base_frame [rad].
        """
        # beam endpoints in robot frame
        # theta already accounts for laser yaw, laser_offset accounts for laser sitting slightly off-center on the robot
        beam_x = self.laser_offset[0] + r * np.cos(theta)		# length B [m] in base_frame
        beam_y = self.laser_offset[1] + r * np.sin(theta)		# length B [m] in base_frame

        # slicing with 0:1 keeps each column as shape (N, 1) instead of (N,) so it broadcasts against the beams
        x = self.particle_cloud[:, 0:1]		# shape (N, 1)
        y = self.particle_cloud[:, 1:2]		# shape (N, 1)
        heading = self.particle_cloud[:, 2:3]		# shape (N, 1)

        # rotate beam endpoints by particle heading and shift by particle (x, y) into map frame
        # (N, 1) combined with (B,) broadcasts to (N, B), so row i holds particle i's endpoint for every beam
        end_x = x + beam_x * np.cos(heading) - beam_y * np.sin(heading)		# shape (N, B) [m] in map frame
        end_y = y + beam_x * np.sin(heading) + beam_y * np.cos(heading)		# shape (N, B) [m] in map frame

        # look up endpoints in likelihood_field
        # off-map beams would only get min_likelihood floor since later add the actual probability
        ix, iy, on_map = self.world_to_cell(end_x, end_y)		# cell indices [cells], each shape (N, B)
        # initialize probabilities with min_likelihood across all, then add the likelihood
        beam_probs = np.full(end_x.shape, self.min_likelihood)
        beam_probs[on_map] += self.likelihood_field[ix[on_map], iy[on_map]]

        # probability of scan = total beam probabilities through multiplication (adding = avg score, which is
        # more prone to thinking a bad scan is good) but multiplying approaches 0 for probabilities, so we
        # add logs of scores. axis=1 sums across each row (the beams), giving one score per particle
        log_scores = np.sum(np.log(beam_probs), axis=1) / self.log_scale		# length N

        # subtract max log_score before exp() so the best particle gets weight = 1.0 before normalizing
        # and nothing goes to 0
        self.particle_weights = np.exp(log_scores - np.max(log_scores))
        self.normalize_particles()

    # this is only if we want to supply an initial pose
    def update_initial_pose(self, msg):
        """Callback to re-initialize the particle filter from an RViz 2D Pose Estimate.

        Args:
            msg (PoseWithCovarianceStamped): Pose estimate message in map frame.
        """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)		# [x (m), y (m), theta (rad)]
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)
        self.publish_particles(msg.header.stamp)		# show the new cloud right away instead of waiting for the robot to move

    def initialize_particle_cloud(self, timestamp=0, xy_theta=None):
        """Initializes particle_cloud with n_particles hypotheses.

        If xy_theta is omitted, places particles uniformly across all free map cells.
        If xy_theta is given (e.g., from an RViz click), spreads particles in a Gaussian
        cloud around xy_theta using init_xy_sigma and init_theta_sigma.

        Args:
            timestamp (builtin_interfaces.msg.Time | int): Message timestamp (default = 0).
            xy_theta (Sequence[float] | None): Mean [x (m), y (m), theta (rad)] pose to
                initialize around, or None for full-map random initialization.
        """
        # create particles randomly
        if xy_theta is None:
            self.particle_cloud = self.random_particles(self.n_particles)
        else:
            # pose estimate, so spread particles around xy_theta (np.tile makes xy_theta the center of every particle)
            self.particle_cloud = self.normal_particles(np.tile(xy_theta, (self.n_particles, 1)))

        # visualization code with copy() added so we dont modify original map_array
        #print("We are reaching this point.")
        #particle_map = self.occupancy_field.map_array.copy()
        #for i in range(self.n_particles):
        #    ix, iy, on_map = self.world_to_cell(self.particle_cloud[i, 0], self.particle_cloud[i, 1])
        #    if on_map:
        #        particle_map[iy, ix] = 200
        #plt.imshow(particle_map, interpolation='nearest')
        #plt.show()
        #print("We are reaching this second point.")

        self.particle_weights = np.ones(self.n_particles)

        # reset odom baseline to where robot is right now so motion before this reset isnt applied to new cloud
        if hasattr(self, 'odom_pose'):
            self.current_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)

        self.normalize_particles()
        self.update_robot_pose()

    def normal_particles(self, centers, xy_sigma=None, theta_sigma=None):
        """Draws one particle [x, y, theta] from a Gaussian around each row of centers.

        Redraws (x, y) up to 100 times for any particle that lands in a wall or off-map.

        Args:
            centers (np.ndarray): M x 3 array of mean [x (m), y (m), theta (rad)] poses, one per particle.
            xy_sigma (float | None): x/y std dev [m] (default = self.init_xy_sigma).
            theta_sigma (float | None): Yaw std dev [rad] (default = self.init_theta_sigma).

        Returns:
            np.ndarray: M x 3 array of sampled [x (m), y (m), theta (rad)] particles.
        """
        # defaults to init spread (like for initial pose), resampling gives this function another sigma
        if xy_sigma is None:
            xy_sigma = self.init_xy_sigma
        if theta_sigma is None:
            theta_sigma = self.init_theta_sigma

        m = len(centers)
        particles = np.empty((m, 3))
        particles[:, :2] = centers[:, :2] + np.random.normal(0, xy_sigma, (m, 2))		# x and y [m]
        # redraw only the particles in a wall or off-map, cap at 100 tries to avoid looping forever
        for _ in range(100):
            bad = ~self.is_free(particles[:, 0], particles[:, 1])		# true for particles that need a redraw
            if not bad.any():
                break
            particles[bad, :2] = centers[bad, :2] + np.random.normal(0, xy_sigma, (np.sum(bad), 2))

        # wrap to [-pi, pi], uses np.arctan2 since angle_normalize uses math.atan2 which only takes one number
        theta = centers[:, 2] + np.random.normal(0, theta_sigma, m)		# [rad]
        particles[:, 2] = np.arctan2(np.sin(theta), np.cos(theta))
        return particles


    def normalize_particles(self):
        """Normalizes particle_weights so all weights sum to 1.0.

        Divides each weight by the total sum. If the total is 0 or non-finite (which
        happens if every particle scores 0), resets all weights to equal 1 / N.
        """
        total = np.sum(self.particle_weights)
        if total > 0 and np.isfinite(total):
            self.particle_weights = self.particle_weights / total
        else:
            self.particle_weights = np.ones(len(self.particle_cloud)) / len(self.particle_cloud)		# uniform 1/N fallback case

    def publish_particles(self, timestamp):
        """Publishes the particle cloud to the particle_cloud topic so RViz can display it.

        Wraps each row of particle_cloud in a Particle object to convert [x, y, theta]
        into a geometry_msgs/Pose via as_pose().

        Args:
            timestamp (builtin_interfaces.msg.Time): Header timestamp for the ParticleCloud msg.
        """
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for i in range(len(self.particle_cloud)):
            x, y, theta = self.particle_cloud[i]
            p = Particle(float(x), float(y), float(theta), float(self.particle_weights[i]))
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        """Callback for incoming laser scans on scan_topic.

        Updates last_scan_timestamp and stores the message in scan_to_process if
        run_loop is done with the previous scan. Also logs (at debug level) how old
        the scan already is when it arrives, to separate that from processing lag.

        Args:
            msg (LaserScan): Incoming laser scan message.
        """
        self.last_scan_timestamp = msg.header.stamp
        # how old the scan already is when it reaches us, before run_loop does anything with it
        arrival_age = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e9		# ns to [s]
        self.get_logger().debug(f"scan arrived {arrival_age:.3f} s after its timestamp")
        # throw away scans until we finish processing the previous scan (run_loop sets scan_to_process = None)
        if self.scan_to_process is None:
            self.scan_to_process = msg

    #
    # HELPER FUNCTIONS! :D
    #

    def build_likelihood_field(self):
        """Constructs self.likelihood_field, a 2D grid matching the map dimensions [width, height].

        Marks wall cells as 1.0 and everything else as 0.0, blurs the grid with a Gaussian
        filter of std dev sigma_hit (converted to cells), and scales the result to
        [0.0, 1.0] so each cell stores how likely a laser beam is to end there.
        """
        m = self.occupancy_field.map
        self.map_resolution = m.info.resolution		# cell size [m/cell]
        # all maps have 0 rad rotation in yaml origin, so only x and y are needed
        self.map_origin_x = m.info.origin.position.x		# bottom-left x origin [m]
        self.map_origin_y = m.info.origin.position.y		# bottom-left y origin [m]

        # m.data is a 1D list (100 = wall, 0 = free, -1 = unknown) starting at bottom-left origin going row by row.
        # reshaping gives [y_cell, x_cell], and transposing lets us index as grid[x_cell, y_cell]
        grid = np.array(m.data).reshape(m.info.height, m.info.width).T		# shape (width, height) [cells]

        # gaussian_filter replaces each cell with a weighted average of nearby cells (blurring kernel via convolution)
        # sigma must be in cells, so divide sigma_hit by map_resolution. mode="constant" treats outside map
        # (for edge pixels) as 0
        walls = (grid > 0).astype(float)
        blurred = gaussian_filter(walls, self.sigma_hit / self.map_resolution, mode="constant")
        # scale so highest value on walls is 1.0 to match our probabilities scaling
        self.likelihood_field = np.minimum(blurred / np.median(blurred[walls > 0]), 1.0)

        # every free cell's (iy, ix) indices, found once here so random_particles can pick from them directly
        # instead of guessing random cells until it hits a free one. map_array is indexed [y, x]
        self.free_iy, self.free_ix = np.nonzero(self.occupancy_field.map_array == 0)

    def world_to_cell(self, x, y):
        """Converts map coordinates (x, y) into grid cell indices (ix, iy).

        Works on both single floats or vectorized numpy arrays.

        Args:
            x (float | np.ndarray): x-coordinate(s) in map frame [m].
            y (float | np.ndarray): y-coordinate(s) in map frame [m].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (ix, iy, on_map) where ix, iy are
                integer cell indices [cells] and on_map is True for points inside the map bounds.
        """
        # get grid cell indices
        ix = np.floor((x - self.map_origin_x) / self.map_resolution).astype(int)
        iy = np.floor((y - self.map_origin_y) / self.map_resolution).astype(int)
        width, height = self.likelihood_field.shape		# map dimensions [cells]
        on_map = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        return ix, iy, on_map

    def get_laser_offset(self, laser_frame):
        """Looks up where the laser is mounted relative to base_frame and returns its (x, y).

        The angles from convert_scan_to_polar_in_robot_frame already account for the direction
        the laser faces, but not for its translational (x, y) offset on the robot.

        Args:
            laser_frame (str): TF frame name of the lidar sensor.

        Returns:
            tuple[float, float]: Laser (x, y) mounting offset [m] relative to base_frame.
        """
        transform = self.transform_helper.tf_buffer.lookup_transform(self.base_frame, laser_frame, Time())
        return (transform.transform.translation.x, transform.transform.translation.y)

    def random_particles(self, n):
        """Returns n particles [x, y, theta] at random free spots on the map.

        Picks n random cells from the free cells saved in build_likelihood_field, converts a
        random sub-cell point inside each cell into meters, and picks a random yaw in [-pi, pi].

        Args:
            n (int): Number of particles to make.

        Returns:
            np.ndarray: n x 3 array of random [x (m), y (m), theta (rad)] particles.
        """
        pick = np.random.randint(0, len(self.free_ix), n)		# index into the free cell list for each particle
        # convert cell to meters, adding np.random.random() [0, 1) to cell to land somewhere inside the cell
        # as opposed to always on some grid
        x = self.map_origin_x + (self.free_ix[pick] + np.random.random(n)) * self.map_resolution		# [m]
        y = self.map_origin_y + (self.free_iy[pick] + np.random.random(n)) * self.map_resolution		# [m]
        theta = np.random.uniform(-math.pi, math.pi, n)		# random heading [rad]
        return np.column_stack((x, y, theta))

    def random_particle(self):
        """Returns one particle [x, y, theta] at a random free spot on the map.

        Returns:
            np.ndarray: Random [x (m), y (m), theta (rad)] particle.
        """
        return self.random_particles(1)[0]

    def is_free(self, x, y):
        """Checks if the point(s) (x, y) are on the map and in free space (0).

        Works on both single floats or vectorized numpy arrays.

        Args:
            x (float | np.ndarray): x-coordinate(s) in map frame [m].
            y (float | np.ndarray): y-coordinate(s) in map frame [m].

        Returns:
            bool | np.ndarray: True where (x, y) is inside map bounds and map_array[iy, ix] == 0.
        """
        ix, iy, on_map = self.world_to_cell(x, y)
        # clip off-map indices onto the map edge so the lookup cant go out of range, on_map then marks them not free
        height, width = self.occupancy_field.map_array.shape		# [cells]
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)
        return on_map & (self.occupancy_field.map_array[iy, ix] == 0)		# 0 = free cell

    def estimate_pose(self):
        """Returns the best guess of the robot's pose as (x, y, theta).

        Finds the highest-weighted particle and takes the weighted average of every
        particle within cluster_radius of it. Averaging the entire cloud would
        fail if the cloud split into two groups (landing in empty space between them),
        while using only the single best particle makes the estimate jump around between updates.

        Returns:
            tuple[float, float, float]: Estimated (x (m), y (m), theta (rad)) pose in map frame.
        """
        # get index of highest weight to find best particle's position
        best = np.argmax(self.particle_weights)
        best_x, best_y, _ = self.particle_cloud[best]		# best particle position

        # true for every particle within cluster_radius of the best one, only those get averaged
        x = self.particle_cloud[:, 0]
        y = self.particle_cloud[:, 1]
        near = np.hypot(x - best_x, y - best_y) < self.cluster_radius		# length N true/false mask
        theta = self.particle_cloud[near, 2]
        w = self.particle_weights[near]

        avg_x = np.sum(w * x[near]) / np.sum(w)		# weighted average x [m]
        avg_y = np.sum(w * y[near]) / np.sum(w)		# weighted average y [m]
        # headings cant be averaged as plain numbers (+179 deg and -179 deg would average to 0 deg,
        # the opposite direction). average them as unit vectors (cos, sin) and take atan2 for the angle
        avg_theta = math.atan2(np.sum(w * np.sin(theta)), np.sum(w * np.cos(theta)))		# [rad]

        return (avg_x, avg_y, avg_theta)

def main(args=None):
    """Initializes ROS2, spins the ParticleFilter node, and shuts down on exit."""
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
