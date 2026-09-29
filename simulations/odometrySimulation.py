import math;

def odoSimEstimation(vLeft, vRight, wheelDistance, time, dt, initialX = 0, initialY = 0, initialAngle = 0):
    # check preconditions
    if(type(vLeft) == float | type(vLeft) == int):
        raise TypeError("vLeft must be a float or an integer");
    elif(type(vRight) == float | type(vRight) == int):
            raise TypeError("vRight must be a float or an integer");
    elif(type(wheelDistance) == float | type(wheelDistance) == int):
            raise TypeError("wheelDistance must be a float or an integer");
    elif(type(time) == float | type(time) == int):
            raise TypeError("d must be a float or an integer");
    elif(type(dt) == float | type(dt) == int):
            raise TypeError("dt must be a float or an integer");
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
    finalX = initialX;
    finalY = initialY;
    # average linear velocity
    v = (vLeft + vRight) / 2;
    # average angular velocity.
    omega = (vRight - vLeft) / wheelDistance;
    
    # the angle yeah
    theta = initialAngle;
    # approximate position using dt
    for i in range(int(time/dt)):
        finalX += v * math.cos(theta) * dt
        finalY += v * math.sin(theta) * dt
        theta += omega * dt
    print(f"final position (x, y, angle): ({round(finalX, 3)}, {round(finalY, 3)}, {round(theta, 3)})");
    return (finalX, finalY, theta);
    

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
    
    # average linear velocity
    v = (vLeft + vRight) / 2;
    # average angular velocity.
    omega = (vRight - vLeft) / wheelDistance;
    # angle relative to 0
    theta = omega * time;
    if(theta == 0):
        finalX = initialX + v * time * math.cos(initialAngle)
        finalY = initialY + v * time * math.sin(initialAngle)
    else:
        # total distance traveled
        s = v * time;
        # radius of the path taken
        r = s / theta
        # after pose
        finalX = initialX + r * (math.sin(theta + initialAngle) - math.sin(initialAngle));
        finalY = initialY + r * (math.cos(initialAngle) - math.cos(theta + initialAngle));
        finalAngle = theta + initialAngle;
    return (finalX, finalY, finalAngle);