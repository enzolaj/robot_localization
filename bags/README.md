# Bag Files

This folder holds both the input bags the filter was developed on (provided with the project) and the output bags that were recorded while our filter ran. All of these bags use the map [`maps/mac_1st_floor_final.yaml`](../maps/mac_1st_floor_final.yaml).

## Input Bags

These were provided with the course starter code and were recorded on a Turtlebot 4 driving through the first floor of the MAC.

## Output Bags

Each output bag was recorded while `pf.py` ran with its default parameters on one of the input bags, with the map server running and the bag played at normal speed with `--clock`:

```bash
ros2 bag record -o <name> --use-sim-time /particle_cloud /tf /tf_static /scan /map /clock
```

`/tf` contains both the bag's odometry and the `map` to `odom` transform broadcast by our filter, so replaying a bag shows exactly where the filter placed the robot at every moment. To keep each bag well under GitHub's file size limit, every `/particle_cloud` message was reduced afterwards to a random 1000 of its 6000 particles. 

| Bag | Input | Start | Result |
| :--- | :--- | :--- | :--- |
| `pf_take1_random_start` | take 1 | particles spread over the whole map | The cloud first settles in the wrong room (~9 s), jumps to a second wrong spot (~18 s), and locks onto the correct pose by ~26 s, where it stays for the rest of the bag. |
| `pf_take1_pose_estimate` | take 1 | 2D Pose Estimate at (0.97 m, 0.13 m, 0.05 rad) | The cloud starts on the robot and tracks it for the entire bag. |
| `pf_take2_random_start` | take 2 | particles spread over the whole map | The cloud stays spread out while the robot is stationary, since the filter only updates after the robot moves, then converges within a few seconds of the robot starting to drive (~10 to 15 s). |

## Playing a Bag

An output bag already contains the filter's results, so do not run `pf.py` alongside it. Start RViz with our configuration first, since the map is published only once at the start of the bag, then play the bag:

```bash
rviz2 -d ~/ros2_ws/src/robot_localization/rviz/turtlebot_bag_files.rviz --ros-args -p use_sim_time:=true
ros2 bag play ~/ros2_ws/src/robot_localization/bags/pf_take1_random_start --clock
```

To run the filter yourself on an input bag instead, follow Section 6.3 of the main [README](../README.md).