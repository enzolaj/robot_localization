# Robot Localization via Monte Carlo Particle Filter

Author Names: Liam B., Enzo S.

For Olin ENGR3590 Computational Introduction to Robotics

This README is a comprehensive report covering both the conceptual math and the programmatic implementation of our Monte Carlo Localization (MCL) particle filter. Unless stated otherwise, the parameters in the tables below are the defaults declared in `pf.py`; a few figures and recordings were made with different values (mainly `n_particles`), though this is noted in line.  All results come from the two Turtlebot 4 bags recorded on the first floor of the MAC, [`bags/macfirst_floor_take_1`](bags/macfirst_floor_take_1) and [`bags/macfirst_floor_take_2`](bags/macfirst_floor_take_2). For our purposes, the parameters were set with these environments in mind as opposed to the Neato Gauntlet World. 

## 1. Project Overview

Given a robot in a known environment, we can poll the robot's sensors to determine what it "sees" around itself; however, there is no direct way from our known environment, or map, to extract the pose of the robot itself. Thus, we simulate thousands of particles and mimic the same sensing the robot experiences as the particles. Through this, we can determine which particles are likely to match our robot's pose, and without directly interacting with the robot, extract said robot's pose through this external algorithmic layer. Picture the robot switched on somewhere on the first floor of the MAC (a building at Olin). It has a map of the building and two sensors: wheel odometry, which reports how far it has driven, and a 2D lidar, which reports how far away the nearest surface is in every direction.  Indepdently, these sensors are unable to provide sufficient information for pose extraction. Using odometry can only describe motion relative to the robot's starting location and the lidar can only describe the surrounding area with minimal ability to differentiate between seemingly similar environments. The goal of this project is to combine the two in a particle filter that keeps track of the robot's pose on a known occupancy grid map in ROS2.

The filter runs as a single node, `pf.py`, with five stages:

- [Particle cloud initialization & likelihood field precomputation](#map-and-init)
- [Motion model (odometry update)](#motion-model)
- [Sensor model (laser scan scoring)](#sensor-model)
- [Pose estimation & the map to odom transform](#pose-estimate)
- [Resampling](#resampling)

External but necesarry to this process, we use 2D Gaussian convolution (`scipy.ndimage.gaussian_filter`) to build the likelihood field, rigid body frame transformations to move scans and particles between frames, log-likelihood summation with a scaling constant (`log_scale`) to score particles, vectorized NumPy array operations and broadcasting to update thousands of particles at once, and a circular mean to average headings.


We decided to work asynchronously which led to us dividing the work for this project during meetings into discrete parts which we would complete before the next meeting. Respectivly we each wrote the code for:

Enzo: 
Liam: particle initation, particle resampling

And similarly worked on the deliverables and README

Enzo:
Liam: README particle initation, README particle initation

[TODO: division of labor paragraph, e.g. who wrote which methods, who tuned parameters on the bag files, who wrote the tests, and how the README was split up.]

### 1.1 Monte Carlo Localization in Brief

Before getting into the code, we will describe the algorithm on its own. A particle is a single guess of the robot's pose, $[x\ (\mathrm{m}),\ y\ (\mathrm{m}),\ \theta\ (\mathrm{rad})]$, in the map frame, and its weight describes how well the most recent laser scan aligns with that guess. The filter keeps thousands of these guesses at once and repeats three steps every time the robot moves far enough:

```
for every update:
    predict:   move every particle by the odometry change, plus noise
    correct:   score every particle by how well the scan lines up with the map from its pose
    resample:  build a new cloud where high-weight particles get more copies
```

Keeping a cloud instead of a single estimate has the advantage of holding several possibilities, or pose guesses, at a time. If, for example, two hallways looked identical to the lidar, the filter could keep a group of particles between both points until more information would arrive that could differentiate the two, unlike single possibility algorithms, like a Kalman filter. With this, the cost of computation is far higher. This required a focus on computational efficiency in our work, as described later. 

Our implementation follows the particle filter and Monte Carlo Localization algorithms described in *Probabilistic Robotics* by Thrun, Burgard, and Fox (ch. 4 and ch. 8), visualized on a high level below in Figure 1.

<p align="center"> <img src="docs/mcl_loop.png" alt="The predict, correct, resample loop"> </p> <p align="center"><em>Figure 1: The MCL loop. [TODO: make this diagram]</em></p>

### 1.2 Architecture

#### 1.2.1 Topics, TF Frames, and Threading

The node subscribes to the laser scan and to RViz's 2D Pose Estimate tool, publishes the particle cloud for visualization, and broadcasts a single transform, `map` to `odom`, which is how ROS learns where the filter thinks the robot is. Table 1 lists every topic and transform the node interacts with.

*Table 1: Particle filter node topics and TF transforms, with when each one is used and its role in the filter.*

| Topic / TF Frame | Direction | Message / Transform Type | When | Role |
| :--- | :--- | :--- | :--- | :--- |
| `/scan` | subscribe | `sensor_msgs/LaserScan` | every scan (~7.7 Hz in the bags) | Stores the newest scan in `scan_to_process` if the filter is free, and updates `last_scan_timestamp` |
| `/initialpose` | subscribe | `geometry_msgs/PoseWithCovarianceStamped` | on RViz 2D Pose Estimate click | Re-initializes the cloud as a Gaussian around the clicked pose |
| `/particle_cloud` | publish | `nav2_msgs/ParticleCloud` | after the cloud is created or updated | Displays every particle's pose and weight in RViz |
| `odom` -> `base_footprint` | TF lookup | `geometry_msgs/TransformStamped` | every `run_loop` tick | Retrieves the robot's odometry pose at the exact time the scan was taken |
| `base_footprint` -> `rplidar_link` | TF lookup | `geometry_msgs/TransformStamped` | per scan | Provides the lidar's (x, y) mounting offset on the robot |
| `map` -> `odom` | TF broadcast | `geometry_msgs/TransformStamped` | 20 Hz timer (+0.1 s postdated) | Corrects odometry drift by placing the odom frame on the map |

The particle cloud is published with `qos_profile_sensor_data`, which uses best effort reliability. Since the cloud is purely a visualization topic that gets replaced per update, the message being dropped, where best effort will not guarantee every message being sent or retransmitted if lost, is largely inconsequential. 

The filter itself does not run inside the scan callback. `scan_received` only stores the newest scan in `scan_to_process`only once the previous scan has been handled, while `loop_wrapper` calls `run_loop` repeatedly on a separate daemon thread. This intentional decoupling prevents a slow filter update from blocking ROS callbacks, meaning scans that arrive while the filter is busy are discarded rather than accumulating in a queue and building delay in communication. Originally, `loop_wrapper` slept for 0.1 s between calls, so a scan that arrived immediately after a check could wait up to 0.1 s before `run_loop` processed it. Reducing the sleep to 0.01 s lowered the median lag over the first 14 s of the bag from approximately 0.21 s to 0.15 s (Section 3.2).

A 20 Hz timer, `pub_latest_transform`, re-broadcasts the latest `map` to `odom` transform  with a hardcoded time stamped 0.1 s after the last scan's timestamp. Postdating, or adding time to the original timestamp, the transform prevents TF extrapolation errors in RViz between laser updates, and deriving the stamp from the scan rather than the wall clock keeps every node on bag time during playback so they are in unison.

One minor tweak for performance was modifying when the particle cloud is published. Initially, the cloud was published after every scan; however, publishing several thousand particles required 55 to 72 ms, which exceeded the cost of the filter update itself, and the cloud only changes when the filter updates. `publish_particles` is now called only after the cloud is created or updated.

Because the bags were recorded on a Turtlebot 4 rather than a Neato, the sensor setup differs from the gauntlet simulation  and previous project in three main ways:

1. The bag publishes both `odom` -> `base_link` and `odom` -> `base_footprint`. The filter tracks the robot through `base_footprint`, but the lidar is attached to `base_link` by a static transform, so `get_laser_offset` resolves the lidar's position through that chain. The resulting offset of about 4 mm is negligible next to the 0.03 m map cell, though this is good practice as the Neato has a non-negligible offset at this resolution.
2. `rplidar_link` is rotated by $\pi/2$ relative to the robot, while the Neato's laser faces backwards, a rotation of $\pi$. `convert_scan_to_polar_in_robot_frame` adds this yaw to every beam angle (Section 2.3.1). If it were omitted, every scan would be rotated relative to the robot, and the filter would favor particles with the wrong heading. One of our unit tests checks this on a Neato scan (Section 4.2).
3.  The rplidar returns 720 beams per scan, about twice the Neato's 361. Since scoring cost grows with the number of beams, this is the main reason for subsampling with `beam_step` (Section 2.3.1).

<p align="center"> <img src="docs/tf_tree.png" alt="TF tree"> </p> <p align="center"><em>Figure 2: TF tree during bag playback, map -> odom -> base_footprint, with the lidar attached through base_link. [TODO: generate with ros2 run tf2_tools view_frames]</em></p>

#### 1.2.2 Filter Parameters

Table 2 below lists every ROS2 parameter declared in `__init__`, its function, and how its value was determined. Since they are declared as ROS parameters, each can be overridden at launch with `--ros-args -p name:=value` for ease of use.

*Table 2: Particle filter parameters, their defaults, tuning effects, and how each value was chosen.*

| Parameter                                   | Value                | Role & Tuning Effect                                                                                                                              | How It Was Chosen                                                                  |
| :------------------------------------------ | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------- |
| `n_particles`                               | 6000 particles       | More particles cover the map more densely, improving the reliability of global localization, at the cost of computation on every update           | Lag measurements: 10000 localizes more reliably but exceeds the ~0.13 s lag target |
| `d_thresh`, `a_thresh`                      | 0.05 m, $\pi/24$ rad | Distance or rotation required before the filter updates; smaller values produce more frequent updates and more frequent resampling                | [TODO]                                                                             |
| `random_resample`                           | 0.2                  | Fraction of particles respawned uniformly over the map on every resample, allowing recovery if the filter converges on the wrong location         | [TODO]                                                                             |
| `resample_xy_sigma`, `resample_theta_sigma` | 0.1 m, 0.1 rad       | Standard deviation of the noise added to each resampled copy; tighter than the initial spread, since resampled particles have already scored well | [TODO]                                                                             |
| `sigma_hit`                                 | 0.3 m                | Width of the blur around walls; a wider blur tolerates more sensor and map error at the cost of precision                                         | [TODO]                                                                             |
| `min_likelihood`                            | 0.01                 | Lower bound on every beam's probability, so a single erroneous reading cannot eliminate a good particle                                           | [TODO]                                                                             |
| `log_scale`                                 | 20                   | Divisor on each particle's summed log score, so ~240 correlated beams are not treated as "240 independent measurements" (Section 2.3.3)           | Measured weight concentration on a real scan, followed by testing on the bag       |
| `beam_step`                                 | 3                    | Uses every third beam (720 -> 240) to reduce update time                                                                                          | Computational speed                                                                |
| `init_xy_sigma`, `init_theta_sigma`         | 0.3 m, 0.3 rad       | Spread of the initial cloud around a 2D Pose Estimate                                                                                             | [TODO]                                                                             |
| `odom_trans_noise`, `odom_rot_noise`        | 0.1 m/m, 0.1 rad/rad | Motion noise as a fraction of distance driven and angle turned                                                                                    | [TODO]                                                                             |
| `cluster_radius`                            | 1.0 m                | Radius around the best particle within which particles are averaged into the pose estimate                                                        | [TODO]                                                                             |

Many of these values have a particular interplay with other functionality of the system - for example, modifying `random_resample` translates to a modification in many of the standard deviations of other distributions and the total particle count. With `random_resample` at 0.2, every update places 1200 of the 6000 particles at random over roughly 1371 m$^2$ of free space, a density of only about 0.9 particles per m$^2$. Once the filter has localized, almost none of these land near the robot, so in practice roughly 4800 particles perform the tracking. 

### 1.3 Data: Bag Files and Map

*Table 3: Bag files used for development and testing.*

| Bag | Length | Topics | Notes |
| :--- | :--- | :--- | :--- |
| [`macfirst_floor_take_1`](bags/macfirst_floor_take_1) | ~93 s | `/scan` (~7.7 Hz, 720 beams), `/tf` (~20 Hz), `/tf_static`, `/odom`, `/cmd_vel` | A lag spike at ~58.9 s appears in every run, indicating it originates in the bag |
| [`macfirst_floor_take_2`](bags/macfirst_floor_take_2) | ~46 s | same as take 1 | The robot is stationary for the first ~11 s |

The map, [`maps/mac_1st_floor_final.yaml`](maps/mac_1st_floor_final.yaml), has a resolution of 0.03 m per cell and approximately 1371 m$^2$ of free space. Consequently, the resulting particle density is low for our selected 6000 particles, amounting to only about 4.4 per m$^2$, which is a large component of the difficulty of global localization from a random cloud (Section 2.1.2). Unknown (gray) cells are treated as empty when building the likelihood field, so any surface the lidar detects in an unmapped region scores the same as open space. Though the map was developed via SLAM tools on an earlier run inside that same bag recording, there still exist tiny scaling differences in the way that the points are interpreted during SLAM to convert to the map. For example, areas that are explored more densely lead to thinner walls as more points exist there and can cleanly extract some depth, similar to the irregular thickness of obstacle walls on the Neato Gauntlet Word. Thus, the scan may not exactly line up 1-1 with the map data, though in our testing this was negligble in delivering a solid pose extraction algorithm.

[TODO: add bags/README.md describing each bag's route and how to play it with]

## 2. Core Localization Subsystems

This section describes the five stages of the filter in the order they execute. For each stage, we cover its purpose, implementation, underlying math, limitations, and related unit tests.

### <a name="map-and-init" id="map-and-init"></a>2.1 Map Representation, Likelihood Field, & Initialization

#### 2.1.1 Precomputing the Likelihood Field

Scoring a particle requires determining, for every laser beam, whether that beam would have terminated on a wall had the robot been at the particle's pose. Tracing every beam through the map on every update would be prohibitively slow, so this computation is performed once, in advance, for every cell. `build_likelihood_field` converts the map into a grid in which each cell stores the likelihood of a beam ending there. Scoring then reduces to a lookup.

The map arrives as `m.data`, a flat list in which 100 denotes a wall, 0 free space, and -1 unknown, ordered row by row from the map's bottom-left origin. Reshaping it to `(height, width)` produces rows of y and columns of x, and transposing it allows indexing as `grid[x_cell, y_cell]`, consistent with the (x, y) convention used throughout the code (and aligned with developer preference). Walls are then marked with `(grid > 0).astype(float)`, which assigns 0 to both free and unknown cells, and blurred with `gaussian_filter` using a standard deviation of `sigma_hit / map_resolution` cells. With `mode="constant"`, everything beyond the edge of the map is treated as empty.

The blurred field must be scaled so that a wall reads as approximately 1.0. Our first version accomplished this by dividing by `np.max(blurred)`, the brightest value on the entire map. This proved to be a significant flaw. One property of a gaussian blur with our kernel size is that, given walls of uneven thickness, the multi-pixel thick walls would influence the nearby pixels twice as much, leading to more intense brightness for thicker walls. Consequently, only thick walls and corners reach the maximum, so thin walls were scaled far lower: on the MAC map, the median wall cell was only about 0.38. A beam landing directly on a thin wall therefore scored lower than a beam landing just near a thick one, and earlier in our unit tests a pose 20 cm from the truth outscored the true pose (`test_true_pose_scores_best` and `test_a_few_bad_beams_dont_change_the_winner` both failed).

Blurring a line one cell thick with a Gaussian of standard deviation $s$ cells distributes its single unit of intensity across the entire kernel, so its peak is approximately

$$
\text{peak}_{\text{thin}} \approx \frac{1}{\sqrt{2\pi}\, s},
$$

which is only about 0.04 at $s = 10$ cells. The edge of a very thick wall, by contrast, reaches 0.5, and the interior of a large solid region reaches 1.0. Before scaling, a wall's intensity depends on how thickly it is drawn in the map, which is unrelated to whether a beam would strike it.

Our fix was to scale by a typical wall rather than the brightest one, and to cap the result at 1:

```python
self.likelihood_field = np.minimum(blurred / np.median(blurred[walls > 0]), 1.0)
```

A typical wall now reads as 1.0, thicker walls saturate at 1.0, and both previously failing tests pass. The cap does carry a cost: near thick walls and corners the field is uniformly 1.0, so slightly misaligned particles in those regions receive identical scores.

<p align="center"> <img src="docs/likelihood_field.png" alt="Likelihood field before and after the median fix"> </p> <p align="center"><em>Figure 3: The likelihood field before (left) and after (right) scaling by the median wall value. Note how the thin walls go from dim to fully bright. [TODO: make with view_likelihood_field.py]</em></p>

We also considered an alternative construction. *Probabilistic Robotics* (ch. 6.4) describes a likelihood field derived from each cell's distance to the nearest wall, which is the approach ROS AMCL uses (`scipy.ndimage.distance_transform_edt` computes it), and which assigns every wall exactly 1.0 by construction. [TODO: say why you kept the blur, e.g. it was your own approach, it was already working with the fix, you wanted to understand the problem instead of swapping methods.] We had explored the distance transform early in development in `test2.py`.

To look up a point $(x, y)$ in the field, `world_to_cell` converts it from meters to cell indices:

$$
i_x = \left\lfloor \frac{x - x_{\text{origin}}}{\text{resolution}} \right\rfloor, \qquad i_y = \left\lfloor \frac{y - y_{\text{origin}}}{\text{resolution}} \right\rfloor
$$

and additionally reports whether the point lies on the map. It operates identically on a single point or on entire NumPy arrays of points, which the laser update relies on.

Precomputation represents a clear tradeoff: the blur runs once at startup over the whole map, whereas scoring performs roughly 6000 particles × 240 beams of lookups on every update. The same function also stores the indices of every free cell (`np.nonzero(map_array == 0)`), so random particles can be drawn directly from free space instead of by repeated trial.

#### 2.1.2 Particle Representation & Cloud Initialization

The cloud is stored as an N × 3 NumPy array with one row $[x\ (\mathrm{m}),\ y\ (\mathrm{m}),\ \theta\ (\mathrm{rad})]$ per particle, and rows are converted to `Particle` objects only when publishing to RViz for ease of use. Representing the cloud as a single array is what allows every other stage to update all particles in one operation.

The cloud can be initialized in two ways:

First, if there is no initial estimate for the robots position the particle cloud is generated by generating an `n_particles` long list of cordinates with a uniform distribution throughout every non-ocupied cell (`free_iy`, `free_ix`). Second if there is an intial guess for where the robot is, the particle cloud is generated using `normal_particles` which generates all `n_particles` around the initial guess with a normal distribution.

When the particle cloud is initalized around a best guess for the robots position, there is only one point around which the particle is generated as oposed to the many during resampling, becasue of this a larger standard deviation is used for the inital guess (`init_xy_sigma`, `init_theta_sigma`) than for the resampling (`resample_xy_sigma`, `resample_theta_sigma`).




[TODO: talk more about this section]
[TODO: include a table that shows how the distribution would matter in map scale prob]

### <a name="motion-model" id="motion-model"></a>2.2 Odometry Motion Model

#### 2.2.1 Update Trigger Thresholds

The filter updates only when `moved_far_enough_to_update` determines that the robot has translated more than `d_thresh` (0.05 m) or rotated more than `a_thresh` (7.5 degrees) since the previous update. Updating on every scan while the robot is stationary would repeatedly resample the same information, and the noise added at each resample would progressively spread the cloud, or collapse it onto whichever particle scored highest, without any new evidence about the robot's location. With this naive implementation, all of these checks run independently, meaning that diagonal movement could travel further along this check without triggering it. 

#### 2.2.2 Coordinate Frame Rotation & Scaled Gaussian Noise

The odometry change $(\Delta x_{\text{odom}}, \Delta y_{\text{odom}}, \Delta\theta)$ is measured in the odom frame and cannot be added directly to each particle's position, because each particle has its own heading $\theta_i$ and the odom frame is rotated relative to the map frame by an unknown amount. If odometry reports a 1 m displacement in +y, the physical meaning is that the robot drove 1 m forward, and a particle facing +x on the map should therefore move 1 m in +x.

`update_particles_with_odom` accordingly proceeds in two steps. First, it rotates the odometry change by the robot's previous odometry heading $\theta_{\text{old}}$ into the robot's body frame, expressed as forward, left, and turn components:

$$
\text{forward} = \cos(\theta_{\text{old}})\Delta x + \sin(\theta_{\text{old}})\Delta y
$$

$$
\text{left} = -\sin(\theta_{\text{old}})\Delta x + \cos(\theta_{\text{old}})\Delta y
$$

$$
\text{turn} = \text{angle\_diff}(\theta_{\text{new}}, \theta_{\text{old}})
$$

These expressions follow from projecting the displacement onto the robot's own axes. At heading $\theta_{\text{old}}$, the robot's forward axis in the odom frame is the unit vector $\hat{u}_f = (\cos\theta_{\text{old}}, \sin\theta_{\text{old}})$, and its left axis, obtained by rotating $\hat{u}_f$ by $+90$ degrees, is $\hat{u}_l = (-\sin\theta_{\text{old}}, \cos\theta_{\text{old}})$. The component of the displacement $(\Delta x, \Delta y)$ along each axis is its dot product with that unit vector, which yields the forward and left equations above. Equivalently, the two equations apply the inverse of the 2D rotation matrix, which for a rotation is its transpose:

$$
\begin{bmatrix} \text{forward} \\ \text{left} \end{bmatrix} = R(\theta_{\text{old}})^{T} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}, \qquad R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

$R(\theta_{\text{old}})$ maps a vector from the robot's frame into the odom frame, so its transpose maps the odom displacement back into the robot's frame. The result no longer depends on how the odom frame is oriented, which is what allows the same motion to be applied to every particle. Using $\theta_{\text{old}}$ treats the motion as if it occurred entirely at the starting heading; since updates occur every 5 cm or 7.5 degrees, the error from ignoring the heading change during the step is small. The turn component needs no projection because a rotation about the vertical axis is the same in every frame, but its difference is computed with `angle_diff`, which wraps the result to $[-\pi, \pi]$ so that crossing $\pm 180°$ is not mistaken for a nearly full rotation.

It then adds zero-mean Gaussian noise to each component, with standard deviations proportional to the distance traveled, $d = \sqrt{\text{forward}^2 + \text{left}^2}$, and the angle turned, $|\text{turn}|$:

$$
\hat{f} = \text{forward} + \mathcal{N}(0,\ k_t d), \qquad \hat{l} = \text{left} + \mathcal{N}(0,\ k_t d), \qquad \hat{\tau} = \text{turn} + \mathcal{N}(0,\ k_r |\text{turn}|),
$$

where $k_t$ is `odom_trans_noise` and $k_r$ is `odom_rot_noise`. Scaling the noise with motion reflects how odometry error arises in practice: wheel slip accumulates with distance, so a long move is less certain than a short one, and a stationary robot should not scatter its particles. Finally, each particle applies the noisy motion along its own heading:

$$
x_i' = x_i + \hat{f}\cos(\theta_i) - \hat{l}\sin(\theta_i), \qquad y_i' = y_i + \hat{f}\sin(\theta_i) + \hat{l}\cos(\theta_i), \qquad \theta_i' = \theta_i + \hat{\tau}
$$

These operations are applied to the entire cloud at once. Passing n to `np.random.normal(0, sigma, n)` draws an independent noise sample for each particle, and entire columns of `particle_cloud` are updated. The helper `angle_normalize` relies on `math.atan2`, which accepts only scalars, so the new headings are instead wrapped to $[-\pi, \pi]$ via `np.arctan2(np.sin(theta), np.cos(theta))` over the whole column.

 Our model is relatively simple, with a single translational and a single rotational noise term. Its assumption is that rotation in place introduces no positional noise. A physical robot rotating in place likely experiences some translation as well, although we did not measure its magnitude, yet still deem it negligible. 

#### 2.2.4 Testing & Limitations

Sign errors and frame mix-ups are both easy to introduce in the motion model and difficult to detect in RViz, so most of the tests disable noise and verify exact values. `test_forward_means_forward_for_each_particle` applies the same 1 m motion (facing +y in the odom frame) to particles facing four different directions, each of which must move 1 m along its own heading; `test_sideways_drift` verifies the lateral term; and `test_turning_across_180_degrees` confirms that a change from 3.0 rad to -3.0 rad is interpreted as a small left turn of about 0.28 rad rather than a turn of -6 rad. Two additional tests verify that noise spreads particles without biasing their mean and that a stationary robot does not scatter them.

Regarding limitations, the noise constants are fixed across robots and environments, and they were set by [TODO:]. They are also coupled to `d_thresh`, since larger steps between updates produce larger noise per update.

### <a name="sensor-model" id="sensor-model"></a>2.3 Laser Sensor Model & Log-Likelihood Scoring

#### 2.3.1 Beam Subsampling & Sensor Frame Transform

Before scoring, `run_loop` discards any beam outside the laser's valid range, $[\text{range\_min}, \text{range\_max})$, and retains every `beam_step`-th beam, reducing the 720 beams to approximately 240. At this point, no additional filtering is required as the zero and inf reading cannot bypass the range check.

By reducing the coverage via subsampling, there is less information being captured. The rplidar's beams are spaced 0.5 degrees apart, and the gap between adjacent beam endpoints increases linearly with range. Nearby walls receive many beams while distant walls receive only a few, meaning nearby walls dominate the score. This is partially desirable, since close-range readings are more accurate; however, in a large open room the distant walls carry the most information about the robot's position, so the beam distribution is important. Despite this, our beam selection is not intelligent and naively iterates uniformly across the indices. 

Each beam $k$ arrives as a range $r_k$ and an angle $\phi_k$, and it must be expressed in the map frame before it can be looked up in the likelihood field. This happens in two steps. First, the beam is converted to a Cartesian point in the robot's frame. `convert_scan_to_polar_in_robot_frame()` already adds the laser's mounting yaw to every angle ($\pi/2$ for the Turtlebot's `rplidar_link`), so only the laser's translational offset $(o_x, o_y)$ from `get_laser_offset()` remains to be added:

$$
b_{x,k} = o_x + r_k\cos\phi_k, \qquad b_{y,k} = o_y + r_k\sin\phi_k \tag{11}
$$

Second, the point is placed on the map from the pose of each particle $i$, $[x_i, y_i, \theta_i]$, by rotating it with $R(\theta_i)$ from Eq. (7) and then shifting it by the particle's position:

$$
\begin{bmatrix} e_{x,ik} \\ e_{y,ik} \end{bmatrix} = \begin{bmatrix} x_i \\ y_i \end{bmatrix} + R(\theta_i) \begin{bmatrix} b_{x,k} \\ b_{y,k} \end{bmatrix} \tag{12}
$$

where $(e_{x,ik}, e_{y,ik})$ is the endpoint of beam $k$ as seen from particle $i$. This is the same rotation used in the motion model, applied in the opposite direction: in Eq. (7), $R(\theta_{\text{old}})^{T}$ moves an odometry displacement into the robot's frame, while in Eq. (12), $R(\theta_i)$ moves a point in the robot's frame out onto the map.

Eq. (12) has to be evaluated for every particle and every beam, which at N = 6000 particles and B = 240 beams is roughly 1.4 million endpoints per update. Rather than looping, we use NumPy broadcasting. The particle columns are sliced with `0:1` (e.g., `particle_cloud[:, 0:1]`) so that they keep the shape (N, 1) instead of collapsing to (N,), while `beam_x` and `beam_y` have shape (B,). When an (N, 1) array is combined with a (B,) array, NumPy stretches each one along the other's axis, producing an (N, B) array in which row $i$ holds every beam endpoint for particle $i$. A single call to `world_to_cell()` and a single indexing operation into `likelihood_field` then score every beam of every particle at once. The tradeoff is memory, since each intermediate (N, B) array holds about 1.4 million 64-bit floats, or roughly 11.5 MB.

#### 2.3.2 Log-Probability Summation & Preventing Underflow

Each beam is first assigned a probability of `min_likelihood` (0.01), and beams that land on the map then add the value of the likelihood field $L$ at their endpoint. This floor protects good particles from a single unexpected reading. Without it, one beam landing on a 0.0 cell, for instance because a person walked past the robot or a chair was sitting somewhere the map does not show, would have a probability of 0, and since the beam probabilities are multiplied together, that one beam would set the weight of an otherwise well-aligned particle to 0.

Assuming each beam is independent, the likelihood of the full scan from particle $i$ is the product of its beam probabilities. We chose a product over an average because an average is too forgiving: a particle with half of its beams in open space can still reach a reasonable average, while a product penalizes every misaligned beam. Multiplying several hundred values less than 1, however, quickly underflows, so we instead sum their logarithms. Including the `log_scale` divisor $T$ (Section 2.3.3), the log weight of particle $i$ is:

$$
\ell_i = \frac{1}{T} \sum_{k=1}^{B} \ln\left(p_{\min} + L(e_{x,ik}, e_{y,ik})\right) \tag{13}
$$

where $p_{\min}$ is `min_likelihood` and $L$ is taken as 0 for endpoints that fall off the map. In code, this is a single `np.sum(np.log(beam_probs), axis=1)` over the (N, B) array of beam probabilities, where `axis=1` sums across each row to give one score per particle. To convert back to a weight, the largest log weight is subtracted before exponentiating:

$$
w_i = \exp\left(\ell_i - \max_j \ell_j\right) \tag{14}
$$

The shift is necessary because of how large these sums become. Each beam contributes between $\ln(0.01) = -4.6$ for a complete miss and $\ln(1.01) \approx 0.01$ for a beam directly on a wall, so at $T = 1$ a particle whose ~240 beams all miss sums to roughly $-1105$. A 64-bit float, meanwhile, rounds `exp()` of anything below about $-745$ to exactly 0.0. Without the shift, every poorly aligned particle, and occasionally every particle in the cloud, would receive a weight of 0, and normalizing would then divide by zero. Subtracting the maximum gives the best particle a weight of $\exp(0) = 1$ and leaves the ratio between any two weights unchanged, since $e^{\ell_i - c} / e^{\ell_j - c} = e^{\ell_i - \ell_j}$ for any constant $c$.

#### 2.3.3 Overconfidence & log_scale

Early on, global localization succeeded on some runs and failed on others with identical code. Measuring the weights showed why: on a real gauntlet scan (119 beams), one update on a random cloud of 3000 particles gave a single particle 100% of the weight, for an effective sample size, $\text{ESS} = 1/\sum_i w_i^2$, of 1.0 out of 3000. The best and tenth-best particles differed by only about 0.36 log units per beam, but over 119 beams this compounds to a weight ratio of roughly $e^{-42}$. The cause is the independence assumption behind Eq. (13): neighboring beams usually strike the same wall, so counting each one as separate evidence makes the filter far too confident (*Probabilistic Robotics*, ch. 6).

Dividing the summed log score by `log_scale`, $T$, raises each likelihood to the power $1/T$, which is roughly equivalent to treating the 240 beams as $240/T$ independent measurements (about 12 at T = 20) and softens the penalty for a complete miss from a factor of 0.01 to $0.01^{1/20} \approx 0.79$. On the same scan, T = 20 raised the ESS to about 42, while a cloud already near the truth stayed concentrated. [TODO: describe how you settled on 20 on the bag.] The `weights_to_use` debug log tracks this during a run (Figure 4).

<p align="center"> <img src="docs/weights_to_use.png" alt="weights_to_use over a run"> </p> <p align="center"><em>Figure 4: weights_to_use over a run of take 1 for log_scale = 1 and log_scale = 20. [TODO: plot from the debug logs]</em></p>

Before adopting `log_scale`, we also tried replacing the product with the sum of cubed beam probabilities from Nav2 AMCL's [`likelihood_field_model.cpp`](https://github.com/ros-navigation/navigation2/blob/main/nav2_amcl/src/sensors/laser/likelihood_field_model.cpp), which its source describes as an "ad-hoc weighting scheme." This overcorrected: the weights became nearly uniform (an ESS of about 2500 out of 3000), so each update pulled a nearby cloud only about 2 cm toward the truth, less than the noise and random particles added at every resample, and the filter lost track halfway through the bag. We assume AMCL avoids this because, by default, it adds neither. The experiment did surface the thin wall problem in Section 2.1.1, since the sum of cubes failed two unit tests with the original max-normalized field.

#### 2.3.4 Limitations

[TODO]

### <a name="pose-estimate" id="pose-estimate"></a>2.4 Pose Estimation & TF Correction

#### 2.4.1 Clustering & Circular Mean

[TODO]

#### 2.4.2 The map to odom Transform

[TODO]

<p align="center"> <img src="docs/map_odom.png" alt="map, odom, and robot frames"> </p> <p align="center"><em>Figure 5: The map, odom, and robot frames in the worked example above. [TODO: draw]</em></p>

### <a name="resampling" id="resampling"></a>2.5 Particle Resampling

#### 2.5.1 Random Exploration and Weighted Copies

When resampleing the particles after weighting their likelihood, we firts randomly place a porportion (`random_resample`) of the total number of particles, `random_particle_count`. This helps prevent patricle degeneracy by stoping our filter from converging on a incorrect state estimate, which is especialy likely at the begining or in an envoriment without alot of variation. The randomly placed particles allow the filter to "escape" these incorrect estimates without severly impacting the filters overall efficency. 

Once the random particles have been placed we generate the particles with postions based off the weights calculated by `update_particles_with_laser`. However because these weights apply to all `n_particles` and we are only generating `n_particles - random_particle_count` weighted particles, we sort the particles by weight and sum all the lowest particle wights untill they equal `random_resample` and the discard thoes particles. We do this insted of scaling all the weights to fit the number of weighted particles as it also helps remove the lowest weighted particles which we dont want to use for resampling. With the final list of wights to use for placing the weighted particles, `weights_to_use`, we generate the weighted particles with a normal distribution using `resample_xy_sigma`, `resample_theta_sigma` as the standard distuributuion.

Then we concatenate the to arrays of particles, the randomly placed and weighted particles, `random_particle_cloud` and `normal_particle_cloud` respictively. Becasue of how we round the particle weights to determine the number of random and weighted particles we are left with some empty slots, which we fill these with the highest weighted particles giving one slot to each in decending order untill all slots are full. Finaly giving all new particles equal weights of 1/`n_particles`.



#### 2.5.2 Testing & Limitations

[TODO]

## 3. System Integration, Performance, & Results

### 3.1 End-to-End Flow of run_loop

Each pass of `run_loop()` first checks whether a scan is waiting in `scan_to_process`. If one is, `get_matching_odom_pose()` asks TF for the robot's odometry pose at the exact time the scan was taken. If that transform is not available yet, `run_loop()` returns and tries again on the next pass, and if the scan is older than everything in the TF buffer, it can never be matched and is dropped. Once matched, the scan is converted to polar coordinates in the robot's frame and subsampled. What happens next depends on the state of the filter: the first scan sets the odometry baseline, the second creates the particle cloud, and every scan after that triggers a full update whenever the robot has moved far enough:

```
update_particles_with_odom()  ->  update_particles_with_laser()  ->  update_robot_pose()  ->  resample_particles()  ->  publish_particles()
```

<p align="center"> <img src="docs/run_loop.png" alt="run_loop flowchart"> </p> <p align="center"><em>Figure 6: Flowchart of run_loop. [TODO: make]</em></p>

### 3.2 Measuring and Reducing Lag

We define lag as the current bag time minus the timestamp of the scan being processed, and log it at debug level for every scan. Our target was about 0.13 s, or one scan period at 7.7 Hz (Section 1.2.2). To find where the time went, we also logged how old each scan was when it arrived, how long the filter update took, and how long publishing took; all of these are written to `~/.ros/log` when the node runs with `--log-level pf:=debug`. The update and publish times come from `time.perf_counter()`, which measures wall time, so they can only be compared directly to lag, which is measured in bag time, when the bag plays at `--rate 1`.

*Table 7: Median lag over the first 14 s of take 1 as we changed the code.*

| Version | Median lag (s) |
| :--- | :--- |
| First round of vectorization | ~0.225 |
| + timing logs | ~0.220 |
| + vectorized resampling | ~0.210 |
| + loop_wrapper sleep 0.1 s -> 0.01 s | ~0.150 |
| + publish only after updates | ~0.090 |

Over these changes, the share of scans the filter processed rose from about 57% to about 92%, and the filter update itself dropped from about 59 ms to about 35 ms. Ironically, the two largest improvements had nothing to do with the filter's math. With a 0.1 s sleep in `loop_wrapper()`, a scan arriving just after a check could wait up to 0.1 s before `run_loop()` saw it, and publishing the particle cloud took 55 to 72 ms, longer than the entire update, even on scans that did not change the cloud; we only found the second by timing each part of the loop separately.

The logs also showed a few irregularities. A lag spike at about 58.9 s appeared in every run regardless of the code version, so it likely originates in the bag rather than the filter. Some arrival ages were slightly negative, which we attribute to `/clock` during bag playback. Roughly every fifth publish also took around 0.1 s; Python's garbage collector is one possible cause, but we can't say with certainty.

Because `--symlink-install` runs `pf.py` exactly as saved, several of our log files also had no record of which code version produced them; logging the parameters and git commit at startup would have avoided this.

<p align="center"> <img src="docs/lag.png" alt="Lag over time"> </p> <p align="center"><em>Figure 7: Lag over time on take 1, before and after the changes in Table 7. [TODO: plot from the logs]</em></p>

### 3.3 Vectorization

Our first working version updated each particle individually inside Python loops. We rewrote `update_particles_with_odom()`, `update_particles_with_laser()`, `estimate_pose()`, `resample_particles()`, `is_free()`, `random_particles()`, and `normal_particles()` to operate on entire NumPy arrays instead. NumPy runs these operations in compiled code over the whole array at once, whereas a Python loop sends every line through the interpreter once per particle. Loops that drew a random cell, checked it, and retried one particle at a time now draw for every particle at once and redraw only the failures, at some cost to readability, which is why most array operations in `pf.py` carry a shape comment such as (N, 1) or (N, B).

[TODO: benchmark table, time per call for the loop version vs. the NumPy version at particle count. maybe small comparison script]

### 3.4 Debugging Tools & Visualization

Following feedback from our previous project, we used the ROS logger instead of print statements. The node logs at `info` level for messages that should always appear and at `debug` level for everything else, and debug output for this node alone can be turned on with `--log-level pf:=debug`. Each debug line in `pf.py` was added to answer a specific question:

- **odom pose**: Is odometry arriving, and are its values reasonable?
- **weights_to_use**: Is the cloud collapsing onto a few particles or staying spread out?
- **filter update took / publish took**: Where is the time going?
- **scan arrived**: How old is a scan before the filter touches it?
- **lag**: Is the filter keeping up with the laser?

RViz was our other main tool. With the particle cloud and the laser scan both displayed on the map, a well-localized filter places the scan points on the map's walls, while a lost filter leaves them floating in open space or cutting through walls. The 2D Pose Estimate tool let us restart the filter from a known pose, which helped separate problems with tracking from problems with global localization. We also used `view_likelihood_field.py` to inspect the likelihood field directly, and the commented-out matplotlib code in `initialize_particle_cloud()` plots the starting particles on the map.

We built the filter in stages before combining them into `pf.py`, and `pf_stage1_particles.py` through `pf_stage4_resample.py` are what remain of that process: particle placement first, followed by the odometry update, laser scoring, and resampling. [TODO: say what you checked at each stage before moving on.]

<p align="center"> <img src="docs/rviz_unlocalized.png" alt="RViz before localization"> </p> <p align="center"><em>Figure 8: RViz during bag playback before the filter has localized. Note the red scan points not lining up with the map's walls. [TODO: move robot_localization/Screenshot from 2026-10-19 00-02-27.png to docs/ and crop]</em></p>

### 3.5 Results & Demonstrations

[TODO: embed GIFs and reference each one in the text.]

- **Global localization on take 1.** [TODO: GIF.] The robot starts near the west end of the first hallway, drives about 5 m east, turns and drives roughly 12 m south down a second hallway, and then turns around and heads back north. [TODO: check this route against the recording.] Several candidate clusters form in different places at first, and over the first several seconds they merge into one. [TODO: confirm and add timing.]
- **Starting from a 2D Pose Estimate.** [TODO: GIF and a sentence on how quickly the cloud tightens.]
- **Take 2.** The robot is stationary for the first ~11 s, so the cloud stays spread out until it moves. [TODO: GIF and what happens after it moves.]
- **A failure.** [TODO: GIF of a run that never localized from a random start]

[TODO: recordings made with different parameters than Table 2 should say so in their captions.]

### 3.6 Limitations & Future Improvements

[TODO: ladd and say which of these came from planning oversights and which from running out of time.]

## 4. Testing & Verification

### 4.1 Testing the Math Without ROS

[TODO]

### 4.2 What Each Test File Covers

*Table 8: Unit test files.*

| File | Tests | What it checks |
| :--- | :--- | :--- |
| `test_map.py` | 8 | Likelihood field indexing and shape, field values near and away from walls, `world_to_cell()`, random particles in free space, both initialization modes, the 100-try cap, resetting the odometry baseline |
| `test_motion.py` | 13 | Forward motion along each particle's heading (4 headings), sideways motion, turning across 180°, noise behavior, the update thresholds |
| `test_sensor.py` | 12 | The true pose scoring best, robustness to blocked and off-map beams, the laser offset, a real gauntlet scan, normalization and its fallback, resampling |
| `test_output.py` | 5 | Pose estimation, circular mean, picking the heavier group, the `map` to `odom` transform, the published message |

Most tests are described in Section 2 alongside the code they check; two that are not cover indexing and an edge case. `test_likelihood_field_is_indexed_x_then_y` uses a map wider than it is tall (10 × 6 cells) with a single wall cell, so swapping x and y anywhere in `build_likelihood_field()` changes the field's shape or moves its peak, and it also checks that `is_free()`, which reads `map_array` with the opposite [y, x] indexing, agrees on where the wall is. `test_guess_off_the_map_still_finishes` initializes the cloud 50 m off the map, so it only returns because `normal_particles()` stops after 100 attempts.

### 4.4 Limits of the Tests

The unit tests check each component on its own and do not run the full filter over a bag; our live runs in RViz filled that role. The defaults in `conftest.py` also differ from those in `pf.py` in several places (e.g., `sigma_hit` is 0.2 m rather than 0.3 m, and `d_thresh` is 0.2 m rather than 0.05 m), so the tests exercise slightly different settings than the ones the node actually runs with.

The gauntlet test also needs a sharper `sigma_hit` of 0.1 m, since the gauntlet map is slightly smaller than the simulated world; at 0.2 m the backward-facing pose wins, and at the `pf.py` default of 0.3 m a pose 1 m to the right does. It's difficult to attribute this solely to `sigma_hit` given the poor map fit, but it does suggest 0.3 m may be blurrier than ideal. Lastly, simulated scans contain no noise, which makes most of the sensor tests easier to pass than real data would; only `test_a_few_bad_beams_dont_change_the_winner` adds clutter, by cutting 15% of the beams short.

## 5. Learning Objectives and Final Takeaways

[TODO: name, reflection]

## 6. How To Run

### 6.1 Prerequisites

The package runs on ROS 2 Jazzy and requires `numpy`, `scipy`, `PyKDL`, and `nav2_msgs` (for the `ParticleCloud` message), along with the helper modules in this package (`occupancy_field.py`, `helper_functions.py`, and `angle_helpers.py`). [TODO: link the course setup guide.]

[TODO: fill in package.xml; the description, maintainer, and license still say TODO]]

### 6.2 Download and Build

Clone the repository into the `src` folder of your ROS 2 workspace, then build and source it from the workspace root:

```bash
cd ~/ros2_ws/src
git clone [TODO: repo url]
cd ~/ros2_ws
colcon build --symlink-install --packages-select robot_localization
source install/setup.bash
```

`--symlink-install` links `pf.py` instead of copying it, so changes to the code take effect without rebuilding.

### 6.3 Running the Particle Filter on a Bag

Start the map server in one terminal:

```bash
ros2 launch robot_localization launch_map_server.py map_yaml:=$HOME/ros2_ws/src/robot_localization/maps/mac_1st_floor_final.yaml
```

In a second terminal, start the filter. Any parameter from Table 2 can be overridden with `-p`, and `--log-level pf:=debug` enables the lag and timing logs:

```bash
ros2 run robot_localization pf.py --ros-args -p use_sim_time:=true
# or, for example
ros2 run robot_localization pf.py --ros-args -p use_sim_time:=true -p n_particles:=3000 --log-level pf:=debug
```

In a third terminal, open RViz with our configuration:

```bash
rviz2 -d ~/ros2_ws/src/robot_localization/rviz/turtlebot_bag_files.rviz --ros-args -p use_sim_time:=true
```

Finally, play a bag:

```bash
ros2 bag play ~/ros2_ws/src/robot_localization/bags/macfirst_floor_take_1 --clock
```

Alternatively, `launch/test_pf.py` starts the map server and the filter together from a single terminal. `use_sim_time` and `--clock` together make every node use the bag's clock instead of the wall clock; without them, scan timestamps and TF lookups do not line up. Once the bag starts, the particle cloud should appear spread over the entire map, then gather into a few clusters and eventually a single one as the robot drives.

### 6.4 Running the Tests

With ROS sourced, run pytest from the package root:

```bash
cd ~/ros2_ws/src/robot_localization
pytest
```

All 38 tests finish in well under a second. `pytest.ini` adds `robot_localization/` to the Python path, so the tests can import `pf` directly.

### 6.5 RViz Visualization

`turtlebot_bag_files.rviz` already contains the displays we used. To set them up manually, add:

- `Map` on `/map`
- `LaserScan` on `/scan`
- `ParticleCloud` on `/particle_cloud`, with reliability set to Best Effort
- `TF`, to view `map` -> `odom` -> `base_footprint`

The 2D Pose Estimate tool in the RViz toolbar publishes to `/initialpose`, which restarts the cloud around the selected pose.

## 7. Tool Usage

[TODO: add]
## 8. References

- S. Thrun, W. Burgard, and D. Fox, *Probabilistic Robotics*, MIT Press, 2005: particle filters (ch. 4), motion models (ch. 5), beam models, the independence assumption, and likelihood fields (ch. 6), Monte Carlo Localization and Augmented MCL (ch. 8).
- Nav2 AMCL likelihood field model: https://github.com/ros-navigation/navigation2/blob/main/nav2_amcl/src/sensors/laser/likelihood_field_model.cpp
- D. Fox, "Adapting the Sample Size in Particle Filters Through KLD-Sampling," *International Journal of Robotics Research*, 2003.
- SciPy documentation for `scipy.ndimage.gaussian_filter()`.
- [TODO: course project page and starter code link]
