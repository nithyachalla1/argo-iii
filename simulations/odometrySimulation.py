#initial pose
initialX = 0.0;
initialY = 0.0;
initialAngle = 0.0;

#distance between the 2 wheels
track_width = 0.6;

# average velocity recorded by the left wheel
v_left = 0.8;
# average velocity recorded by the right wheel
v_right = 1.2;

# the time step; delta time
dt = 0.1;
# how long the simulation is run
simulation_time = 10.0;


# 
def odoSim():
    # average linear velocity
    v = (v_left + v_right) / 2;
    # average angular velocity. right is positive, left is negtative
    omega = (v_right - v_left) / track_width;
    # angular distance traveled
    theta = omega * simulation_time;
    # total distance traveled
    s = v * simulation_time;
    # radius of the path taken
    r = s/theta
    # after pose
    finalX = initialX + (r-r*cos(theta));
    finalY = initialY + r*sin(theta);
    finalAngle = initialAngle+theta;
