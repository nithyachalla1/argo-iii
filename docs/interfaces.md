# A.R.G.O. Software Interfaces

These interfaces are preliminary and may change as the rover design is finalized.

| Data                   | Topic                     | ROS Message                  |
| ---------------------- | ------------------------- | ---------------------------- |
| Rover velocity command | `/cmd_vel`                | `geometry_msgs/msg/Twist`    |
| Rover odometry         | `/odom`                   | `nav_msgs/msg/Odometry`      |
| Wheel states           | `/joint_states`           | `sensor_msgs/msg/JointState` |
| IMU data               | `/imu/data`               | `sensor_msgs/msg/Imu`        |
| Left camera            | `/camera/left/image_raw`  | `sensor_msgs/msg/Image`      |
| Right camera           | `/camera/right/image_raw` | `sensor_msgs/msg/Image`      |

## Initial Coordinate Frames

```text
map
 |
odom
 |
base_link
 |-- camera_link
 |-- imu_link
 |-- wheel links
```

## Important Rule

Do not create custom ROS messages unless standard ROS messages cannot reasonably represent the data we need.
