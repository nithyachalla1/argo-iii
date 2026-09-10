# A.R.G.O. Software Architecture

## High-Level System

```text
Cameras
   |
   v
Perception
   |
   v
Obstacle / Terrain Map
   |
   v
Navigation
   |
   | /cmd_vel
   v
Controls
   |
   v
Motors

Encoders ----\
              > Localization ---> Navigation
IMU --------/
```

## Software Areas

### Perception

Responsible for understanding the rover's surroundings.

Initial responsibilities:

* Camera acquisition
* Camera calibration
* Stereo/depth processing
* Obstacle detection

### Localization

Responsible for estimating rover motion and pose.

Initial responsibilities:

* Wheel odometry
* IMU integration
* Coordinate frames

Later:

* Sensor fusion
* Visual odometry
* SLAM

### Navigation

Responsible for deciding how the rover should move toward a goal.

Initial responsibilities:

* Goal handling
* Obstacle-aware planning
* Costmaps
* Velocity command generation

### Controls

Responsible for converting requested rover motion into motor commands.

Initial responsibilities:

* Rover kinematics
* Motor interface
* Wheel velocity control
* Safety stop behavior

### Robot Description / Simulation

Responsible for representing A.R.G.O. digitally.

Initial responsibilities:

* URDF/Xacro
* Rover geometry
* Coordinate frames
* RViz
* Simulation

## Design Rule

Subsystems should communicate through documented ROS interfaces rather than depending directly on each other's internal implementation.

