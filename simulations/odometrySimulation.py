import math;
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
def odoSimEstimation(vLeft, vRight, wheelDistance, time, dt, initialX = 0, initialY = 0, initialAngle = 0):
    # check preconditions
    if(type(vLeft) != float and type(vLeft) != int):
        raise TypeError("vLeft must be a float or an integer");
    elif(type(vRight) != float and type(vRight) != int):
            raise TypeError("vRight must be a float or an integer");
    elif(type(wheelDistance) != float and type(wheelDistance) != int):
            raise TypeError("wheelDistance must be a float or an integer");
    elif(type(time) != float and type(time) != int):
            raise TypeError("d must be a float or an integer");
    elif(type(dt) != float and type(dt) != int and dt == 0):
            raise TypeError("dt must be a non-zero float or an integer");
    elif(type(initialX) != float and type(initialX) != int):
            raise TypeError("initialX must be a float or an integer");
    elif(type(initialY) != float and type(initialY) != int):
            raise TypeError("initialY must be a float or an integer");
    elif(type(initialAngle) != float and type(initialAngle) != int):
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
    print(type(vLeft)== int)
    if(type(vLeft) != float and type(vLeft) != int):
        raise TypeError("vLeft must be a float or an integer");
    elif(type(vRight) != float and type(vRight) != int):
            raise TypeError("vRight must be a float or an integer");
    elif(type(wheelDistance) != float and type(wheelDistance) != int):
            raise TypeError("wheelDistance must be a float or an integer");
    elif(type(time) != float and type(time) != int):
            raise TypeError("time must be a float or an integer");
    elif(type(initialX) != float and type(initialX) != int):
            raise TypeError("initialX must be a float or an integer");
    elif(type(initialY) != float and type(initialY) != int):
            raise TypeError("initialY must be a float or an integer");
    elif(type(initialAngle) != float and type(initialAngle) != int):
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
        finalAngle = initialAngle
    else:
        # total distance traveled
        s = v * time;
        # radius of the path taken
        r = s / theta
        # after pose
        finalX = initialX + r * (math.sin(theta + initialAngle) - math.sin(initialAngle));
        finalY = initialY + r * (math.cos(initialAngle) - math.cos(theta + initialAngle));
        finalAngle = theta + initialAngle;
    _createPlot(v, omega, time, initialX, initialY, initialAngle, finalX, finalY);
    return (finalX, finalY, finalAngle);

def _createPlot(v, omega, time, initialX, initialY, initialAngle, finalX, finalY):
    t = np.linspace(0, time, 400)
    # angle relative to 0
    theta = omega * t;

    if(omega == 0):
        x = initialX + v * t * math.cos(initialAngle)
        y = initialY + v * t * math.sin(initialAngle)
    else:
        # The radius is constant for a constant pair of wheel velocities.
        radius = v / omega
        x = initialX + radius * (np.sin(theta + initialAngle) - math.sin(initialAngle))
        y = initialY + radius * (math.cos(initialAngle) - np.cos(theta + initialAngle))

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    gradient = LinearSegmentedColormap.from_list("start_to_end", ["#1976d2", "#d32f2f"])
    path = ax.scatter(
        x,
        y,
        c=t,
        cmap=gradient,
        norm=Normalize(vmin=0, vmax=max(time, 1e-12)),
        s=18,
        zorder=2,
    )
    ax.plot(x, y, color="0.35", linewidth=1, alpha=0.45, zorder=1)
    colorbar = fig.colorbar(path, ax=ax, label="progress through simulation")
    colorbar.set_ticks([0, time])
    colorbar.set_ticklabels(["start", "end"])

    ax.scatter(initialX, initialY, color="#1976d2", s=90, label="start", zorder=4)
    ax.scatter(finalX, finalY, color="#d32f2f", marker="*", s=150, label="end", zorder=4)

    ax.annotate("start", (initialX, initialY), xytext=(8, 8), textcoords="offset points")
    ax.annotate("end", (finalX, finalY), xytext=(8, 8), textcoords="offset points")
    ax.set_title("Robot odometry trajectory")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    ax.margins(0.15)
    plt.show()
    return ax