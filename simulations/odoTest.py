from odometrySimulation import odometrySimulation
import math;

def test(vLeft, vRight, wheelDistance, time, initialX, initialY, initialAngle, expected):
    val = odometrySimulation(vLeft, vRight, wheelDistance, time, initialX, initialY, initialAngle);
    print();
    print(f"expected position (x, y, angle): ({round(expected[0], 3)}, {round(expected[1], 3)}, {round(expected[2], 3)})");
    print(f"actual position (x, y, angle): ({round(val[0], 3)}, {round(val[1], 3)}, {round(val[2], 3)})");

    passed = roughApprox(val, expected, 0.001)
    print("test "+ "passed" if roughApprox(val, expected, 0.001) else "**FAILED**");
def roughApprox(a, b, margin):
    
    return (abs(a[0] - b[0]) <= margin) & (abs(a[1] - b[1]) <= margin) & (abs(a[2] - b[2]) <= margin);
test(0.8, 1.2, 0.6, 10.0, 0.0,  0.0, 0.0, (0.561,  0.109,  6.667));
test(0.8, 1.2, 0.6,  2.0,  2.0,  1.0, math.pi / 2, ( 0.853,  2.458,  2.904));
