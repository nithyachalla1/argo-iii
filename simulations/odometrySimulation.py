import math;

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

# runs a simulation based on dead-wheel data to track position.
# pre: wheelDistance > 0, simulation time >= 0.
# post: returns the estimated final position of the robot
def odometrySimulation(vLeft, vRight, wheelDistance, time, initialX = 0, initialY = 0, initialAngle = 0):
    # check preconditions
    if(type(vLeft) == float | type(vLeft) == int):
        raise TypeError("vLeft must be a float or an integer");
    elif(type(vRight) == float | type(vRight) == int):
            raise TypeError("vRight must be a float or an integer");
    elif(type(wheelDistance) == float | type(wheelDistance) == int):
            raise TypeError("wheelDistance must be a float or an integer");
    elif(type(time) == float | type(time) == int):
            raise TypeError("time must be a float or an integer");
    elif(type(initialX) == float | type(initialX) == int):
            raise TypeError("initialX must be a float or an integer");
    elif(type(initialY) == float | type(initialY) == int):
            raise TypeError("initialY must be a float or an integer");
    elif(type(initialAngle) == float | type(initialAngle) == int):
            raise TypeError("initialAngle must be a float or an integer");
    elif(wheelDistance <= 0):
        raise ValueError("wheelDistance must be greater than 0");
    elif(time < 0):
        raise ValueError("time must be 0 or greater");
    print(f"intial position (x, y, angle): ({round(initialX, 3)}, {round(initialY, 3)}, {round(initialAngle, 3)})\n"
        + f"track width: {wheelDistance}\n"
        + f"velocity (left, right): ({vLeft}, {vRight})\n" 
        + f"duration: {time}\n")
    # average linear velocity
    v = (vLeft + vRight) / 2;
    # average angular velocity. right is positive, left is negtative
    omega = (vRight - vLeft) / wheelDistance;
    # angular distance traveled
    theta = omega * time;
    # total distance traveled
    s = v * time;
    # radius of the path taken
    r = s / theta
    # after pose
    finalX = initialX + (r - r * math.cos(theta));
    finalY = initialY + r * math.sin(theta);
    finalAngle = initialAngle + theta;
    print(f"final position (x, y, angle): ({round(finalX, 3)}, {round(finalY, 3)}, {round(finalAngle, 3)})");
    return (finalX, finalY, finalAngle);
odometrySimulation(v_left, v_right, track_width, simulation_time, initialX, initialY, initialAngle);