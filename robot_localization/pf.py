#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

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
import random
from scipy.ndimage import gaussian_filter   # blurs the map into the likelihood field

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of KeyboardInterruptthe hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

    # the filter keeps all of its particles together in one numpy array, which makes the math easier.
    # this class is used to turn a single row of that array into a Pose when publishing.

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

            particle_cloud is stored as an N x 3 numpy array, one row [x, y, theta] per particle, in the map frame
            particle_weights: one weight per particle, in the same order as particle_cloud, adding up to 1
            likelihood_field: a blurred copy of the map, indexed [x cell, y cell], that is high on and near walls
            laser_offset: the (x, y) position of the laser on the robot, relative to its center
    """
    def __init__(self):
        super().__init__('pf')
        self.base_frame = "base_footprint"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from

        #initilaze particle field
        self.n_particles = self.declare_parameter("n_particles", 300).value          # the number of particles to use

        #resample particle field
        self.d_thresh = self.declare_parameter("d_thresh", 0.2).value                # the amount of linear movement before performing an update
        self.a_thresh = self.declare_parameter("a_thresh", math.pi/6).value        # the amount of angular movement before performing an update

        self.random_resample = self.declare_parameter("random_resample", 0.1).value # porportion of particles that are randomly placed to prevent particle degenecary

        # scoring laser scans
        self.sigma_hit = self.declare_parameter("sigma_hit", 0.1).value              # [m] how far the blur spreads out from each wall
        self.min_likelihood = self.declare_parameter("min_likelihood", 0.01).value   # every beam scores at least this, so one odd reading can't sink a good particle
        self.beam_step = self.declare_parameter("beam_step", 2).value                # use every 2nd beam, which is plenty and twice as fast
        # the starting cloud around a 2d pose estimate click
        self.init_xy_sigma = self.declare_parameter("init_xy_sigma", 0.3).value      # [m] spread of the cloud around the click
        self.init_theta_sigma = self.declare_parameter("init_theta_sigma", 0.3).value  # [rad] spread of the headings
        # odometry noise, as a fraction of how far the robot drove or turned since the last update
        self.odom_trans_noise = self.declare_parameter("odom_trans_noise", 0.1).value
        self.odom_rot_noise = self.declare_parameter("odom_rot_noise", 0.1).value
         # pose estimate
        self.cluster_radius = self.declare_parameter("cluster_radius", 0.5).value    # [m] particles this close to the best one go into the pose estimate

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        self.particle_cloud = []
        self.particle_weights = []

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)

        # build the likelihood field once up front, so scoring a scan later only takes lookups
        self.build_likelihood_field()

        # we are using a thread to work around single threaded execution bottleneck
        # (daemon=True lets the program exit on ctrl+c, since this thread never finishes on its own)
        thread = Thread(target=self.loop_wrapper, daemon=True)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)


    def pub_latest_transform(self):
        """ This function takes care of sending out the map to odom transform """
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ This is the main run_loop of our particle filter.  It checks to see if
            any scans are ready and to be processed and will call several helper
            functions to complete the processing.

            You do not need to modify this function, but it is helpful to understand it.

            Before the particles are updated, the scan is trimmed down to the readings worth using,
            and the laser's position on the robot is looked up so the beams can be placed correctly.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return

        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
        # print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))

        # keep every beam_step-th beam, and skip readings outside the range the laser reports as valid.
        good_r = []
        good_theta = []
        for k in range(0, len(r), self.beam_step):
            if msg.range_min <= r[k] < msg.range_max:
                good_r.append(r[k])
                good_theta.append(theta[k])
        r = np.array(good_r)
        theta = np.array(good_theta)

        # where the laser sits on the robot 
        self.laser_offset = self.get_laser_offset(msg.header.frame_id)

        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif len(self.particle_cloud) == 0:
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            # we have moved far enough to do an update!
            self.update_particles_with_odom()    # update based on odometry
            self.update_particles_with_laser(r, theta)   # update based on laser scan
            self.update_robot_pose()                # update robot's pose based on particles
            self.resample_particles()               # resample particles to focus on areas of high density
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg.header.stamp)

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        """ Returns True once the robot has driven more than d_thresh or turned more than a_thresh
            since the last update. The turn goes through angle_diff, so swinging from 179 degrees to
            -179 degrees counts as a 2 degree turn rather than 358. """
        return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
               math.fabs(self.transform_helper.angle_diff(new_odom_xy_theta[2], self.current_odom_xy_theta[2])) > self.a_thresh


    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose (i.e. the mode of the distribution)

            This uses a blend of the two: it starts from the most likely particle and averages it
            with the particles around it (see estimate_pose).
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        x, y, theta = self.estimate_pose()
        self.robot_pose = Particle(x, y, theta).as_pose()
        if hasattr(self, 'odom_pose'):
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

    def update_particles_with_odom(self):
        """ Update the particles using the newly given odometry pose.
            The function computes the value delta which is a tuple (x,y,theta)
            that indicates the change in position and angle between the odometry
            when the particles were last updated and the current odometry.

            Every particle is then moved by that same change. Since delta is measured in the odom
            frame, it is first rewritten as how far the robot drove forward and how far it drifted
            to its left. That motion is then applied along each particle's own heading, with a little
            random noise so the particles don't all move in lockstep.
        """
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        # modify particles using delta
        # rotate delta by the robot's old heading to see the move from the robot's point of view:
        # how far it went forward, and how far it went to its left.
        old_heading = old_odom_xy_theta[2]
        forward = math.cos(old_heading) * delta[0] + math.sin(old_heading) * delta[1]
        left = -math.sin(old_heading) * delta[0] + math.cos(old_heading) * delta[1]
        turn = self.transform_helper.angle_diff(new_odom_xy_theta[2], old_heading)

        # odometry gets less reliable the further the robot goes, so the noise grows with the size of the move
        distance = math.hypot(forward, left)
        trans_noise = self.odom_trans_noise * distance
        rot_noise = self.odom_rot_noise * abs(turn)

        for i in range(len(self.particle_cloud)):
            x, y, heading = self.particle_cloud[i]

            # each particle gets its own slightly different version of the move
            noisy_forward = forward + np.random.normal(0, trans_noise)
            noisy_left = left + np.random.normal(0, trans_noise)
            noisy_turn = turn + np.random.normal(0, rot_noise)

            # forward for this particle means along its own heading, so rotate the move by that heading
            x = x + noisy_forward * math.cos(heading) - noisy_left * math.sin(heading)
            y = y + noisy_forward * math.sin(heading) + noisy_left * math.cos(heading)
            heading = self.transform_helper.angle_normalize(heading + noisy_turn)

            self.particle_cloud[i] = [x, y, heading]

    def resample_particles(self):

        random_particle_count = math.floor(self.n_particles * self.random_resample)
        random_particle_cloud = np.zeros([random_particle_count, 3])

        for i in range(random_particle_count):
            random_particle_cloud[i, :] = self.random_particle()

        normal_particle_count = self.n_particles - random_particle_count
        normal_particle_cloud = np.zeros([normal_particle_count, 3])


        particle_weights_sort = np.array(self.particle_weights).argsort()
        print(np.array(self.particle_weights))
        sorted_weights = np.array(self.particle_weights)[particle_weights_sort[::-1]]
        sorted_particles = self.particle_cloud[particle_weights_sort[::-1]]

        total_normal_weights = normal_particle_count/self.n_particles

        weights_to_use = 0
        #  find how many of the weights it takes to make the number of normaly
        #  distributed particles

        while sum(sorted_weights[:sorted_weights.index(weights_to_use)]) > total_normal_weights:
            weights_to_use += 1

        normal_weights = sorted_weights[:(weights_to_use-1)]
        count = 0
        for j in range(weights_to_use):
            for k in range(normal_weights[j] * normal_particle_count):
                normal_particle_cloud[count, :] = self.normal_particle(sorted_particles[j, :])
                count += 1


        #while normal_particle_cloud[-1 :] == np.array([0, 0, 0]):
            #for j in range(math.floor)




        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample in helper_functions.py.
        """
        # make sure the distribution is normalized
        self.normalize_particles()
        # TODO: fill out the rest of the implementation

    def update_particles_with_laser(self, r, theta):
        """ Updates the particle weights in response to the scan data
            r: the distance readings to obstacles
            theta: the angle relative to the robot frame for each corresponding reading

            For each particle, the scan is laid on the map as if the robot were standing at that
            particle, and every beam endpoint is looked up in the likelihood field to see how close it
            lands to a wall. Particles whose scans line up with the walls get the highest weights.
        """
        # where each beam ends, measured from the center of the robot. theta already accounts for the
        # direction the laser faces; laser_offset accounts for it sitting a little off center.
        beam_x = self.laser_offset[0] + r * np.cos(theta)
        beam_y = self.laser_offset[1] + r * np.sin(theta)

        n = len(self.particle_cloud)
        log_scores = np.zeros(n)
        for i in range(n):
            x, y, heading = self.particle_cloud[i]

            # move the beam endpoints to where they'd land if the robot were at this particle: rotate
            # them by the particle's heading, then shift them by its position. end_x and end_y hold one value per beam.
            end_x = x + beam_x * math.cos(heading) - beam_y * math.sin(heading)
            end_y = y + beam_x * math.sin(heading) + beam_y * math.cos(heading)

            # look up each endpoint in the likelihood field; endpoints off the map only get the minimum
            ix, iy, on_map = self.world_to_cell(end_x, end_y)
            beam_probs = np.full(len(r), self.min_likelihood)
            beam_probs[on_map] += self.likelihood_field[ix[on_map], iy[on_map]]

            # the likelihood of the whole scan is every beam's likelihood multiplied together. multiplying
            # hundreds of small numbers rounds down to 0, so we add their logs instead, which ranks the
            # particles the same way.
            log_scores[i] = np.sum(np.log(beam_probs))

        # this was something i saw: subtracting the best score first gives the best particle
        # a weight of exactly 1 before normalizing, so nothing rounds to 0 by accident.
        self.particle_weights = np.exp(log_scores - np.max(log_scores))
        self.normalize_particles()

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)

    def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is omitted, the odometry will be used

            Without a guess, which is the case at startup, the particles are scattered across every open
            cell of the map with random headings, since the robot could be anywhere. With a guess, such as a
            click from 2D Pose Estimate, the particles are drawn from a normal distribution around that x, y,
            and heading. A particle that lands in a wall or off the map is drawn again, so the cloud always
            has n_particles, all in open space.
        """
        # if xy_theta is None:
        #     xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        self.particle_cloud = np.zeros((self.n_particles, 3))
        # create particles
        if xy_theta is None:
            for i in range(self.n_particles):
                self.particle_cloud[i, :] = self.random_particle()
        

        # visualization code, the copy was added so as to not modify original map
        # particle_map = self.occupancy_field.map_array.copy()
        # for i in range(self.n_particles):
        #     ix, iy, on_map = self.world_to_cell(self.particle_cloud[i, 0], self.particle_cloud[i, 1])
        #     if on_map:
        #         particle_map[iy, ix] = 200
        # plt.imshow(particle_map, interpolation='nearest')
        # plt.show()

        self.particle_weights = np.ones(self.n_particles)

        # measure the next move from where the robot is now, so anything it did before this reset
        # isn't applied to the new cloud.
        if hasattr(self, 'odom_pose'):
            self.current_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)

        self.normalize_particles()
        self.update_robot_pose()

    def normal_particle(self, xy_theta):

        x = np.random.normal(xy_theta[0], self.init_xy_sigma)
        y = np.random.normal(xy_theta[1], self.init_xy_sigma)
        # draw again if this spot is a wall or off the map. the cap on tries keeps a click far
        # outside the map from looping forever.
        tries = 0
        while not self.is_free(x, y) and tries < 100:
            x = np.random.normal(xy_theta[0], self.init_xy_sigma)
            y = np.random.normal(xy_theta[1], self.init_xy_sigma)
            tries += 1
        t = self.transform_helper.angle_normalize(np.random.normal(xy_theta[2], self.init_theta_sigma))
        return [x, y, t]



    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0)

            Each weight is divided by the total. If the total is zero or not a real number, which happens
            when every particle scores 0, the weights are reset to all equal instead. """
        total = np.sum(self.particle_weights)
        if total > 0 and np.isfinite(total):
            self.particle_weights = self.particle_weights / total
        else:
            self.particle_weights = np.ones(len(self.particle_cloud)) / len(self.particle_cloud)

    def publish_particles(self, timestamp):
        """ Publishes the particle cloud so rviz can show it. Each row of particle_cloud is wrapped in a
            Particle so its as_pose() can build the message. """
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for i in range(len(self.particle_cloud)):
            x, y, theta = self.particle_cloud[i]
            p = Particle(float(x), float(y), float(theta), float(self.particle_weights[i]))
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop
        if self.scan_to_process is None:
            self.scan_to_process = msg

    # helper functions
    def build_likelihood_field(self):
        """ Builds the likelihood field - a grid the same size as the map, where each cell is how likely
            a laser beam is to end there, where walls are marked 1 and everything else 0, and the result is
            blurred for more information for our algorithm. """
        m = self.occupancy_field.map
        self.map_resolution = m.info.resolution
        # all of our maps have a rotation of 0 (the third origin value in the yaml), so only x and y are needed
        self.map_origin_x = m.info.origin.position.x
        self.map_origin_y = m.info.origin.position.y

        # the map arrives as one long list: 100 for a wall, 0 for free space, -1 for unknown. it starts at
        # the map origin in the bottom left corner and goes row by row. reshaping gives rows of y and
        # columns of x, and transposing lets us index it as grid[x cell, y cell].
        grid = np.array(m.data).reshape(m.info.height, m.info.width).T

        # gaussian_filter replaces each cell with a weighted average of the cells around it, where closer
        # cells count more. its sigma is measured in cells, so sigma_hit gets converted from meters.
        # mode="constant" treats everything outside the map as empty.
        walls = (grid > 0).astype(float)
        blurred = gaussian_filter(walls, self.sigma_hit / self.map_resolution, mode="constant")
        # scale so the highest value is 1
        self.likelihood_field = blurred / np.max(blurred)

    def world_to_cell(self, x, y):
        """ Converts map coordinates in meters into grid cell indices. on_map is True for points that fall
            inside the map. Works on single values or whole numpy arrays. """
        ix = np.floor((x - self.map_origin_x) / self.map_resolution).astype(int)
        iy = np.floor((y - self.map_origin_y) / self.map_resolution).astype(int)
        width, height = self.likelihood_field.shape
        on_map = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        return ix, iy, on_map

    def get_laser_offset(self, laser_frame):
        """ Looks up where the laser is mounted relative to the center of the robot and returns its (x, y).
            The angles from convert_scan_to_polar_in_robot_frame already account for the direction the
            laser faces, but not for where it sits on the robot. """
        transform = self.transform_helper.tf_buffer.lookup_transform(self.base_frame, laser_frame, Time())
        return (transform.transform.translation.x, transform.transform.translation.y)

    def random_particle(self):
        """ Returns one particle [x, y, theta] at a random open spot on the map, facing a random direction.
            Random cells are picked until one is free, then a random point inside that cell is converted
            to meters. """
        map_width = self.occupancy_field.map.info.width
        map_height = self.occupancy_field.map.info.height
        x = random.randint(0, map_width - 1)
        y = random.randint(0, map_height - 1)
        # map_array holds the raw map values with rows as y and columns as x. only 0 is free space,
        # so this skips both walls (1/home/liam/ros2_ws/src/robot_localization/maps/gauntlet.yaml00) and unknown areas outside the walls (-1).
        while self.occupancy_field.map_array[y, x] != 0:
            x = random.randint(0, map_width - 1)
            y = random.randint(0, map_height - 1)
        t = random.uniform(-math.pi, math.pi)
        # turn the cell into meters, adding random.random() to land somewhere inside the cell rather than on its corner
        x_meters = self.map_origin_x + (x + random.random()) * self.map_resolution
        y_meters = self.map_origin_y + (y + random.random()) * self.map_resolution
        return [x_meters, y_meters, t]

    def is_free(self, x, y):
        """ True if the point (x, y), in meters, is on the map and in free space """
        ix, iy, on_map = self.world_to_cell(x, y)
        return bool(on_map) and self.occupancy_field.map_array[iy, ix] == 0

    def estimate_pose(self):
        """ Returns the best guess of the robot's pose as (x, y, theta). It finds the highest weighted
            particle and takes the weighted average of every particle within cluster_radius of it.
            Averaging the entire cloud would fail if it split into two groups, because the average would
            land in the empty space between them. Using only the single best particle would make the
            estimate jump around from one update to the next. """
        best = np.argmax(self.particle_weights)
        best_x, best_y, _ = self.particle_cloud[best]

        total_weight = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_cos = 0.0
        sum_sin = 0.0
        for i in range(len(self.particle_cloud)):
            x, y, theta = self.particle_cloud[i]
            w = self.particle_weights[i]
            if math.hypot(x - best_x, y - best_y) < self.cluster_radius:
                total_weight += w
                sum_x += w * x
                sum_y += w * y
                # headings can't be averaged as plain numbers: 179 and -179 degrees would average to 0,
                # the opposite direction. averaging them as arrows (cos, sin) and taking the direction of
                # the result gives 180, which is right.
                sum_cos += w * math.cos(theta)
                sum_sin += w * math.sin(theta)

        return (sum_x / total_weight, sum_y / total_weight, math.atan2(sum_sin, sum_cos))

def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
