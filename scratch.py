import numpy as np
import quaternion
import casadi as cs
import liecasadi as liecs



def qqmult(q1, q2):
    """multiply two quaternions or a quaternion and a vector"""
    M = q1.shape[1]
    a = liecs.SO3(q1[:, 0])
    b = liecs.SO3(q2[:, 0])
    q3 = (a * b).xyzw
    for i in range(1, M):
        a = liecs.SO3(q1[:, i])
        b = liecs.SO3(q2[:, i])
        q3 = cs.horzcat(q3, (a * b).xyzw)
    return q3
J = cs.DM(np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]]))
a = cs.MX.sym('a', 3)
v1 = cs.DM(np.array([[1, 10, 1, 10], [2, 20, 2, 20], [3, 30, 3, 30]])).T
v2 = cs.DM(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
v3 = cs.DM(np.array([1, 2, 3, 4]))

q = cs.MX.sym('q', 4, 1)
v = cs.MX.sym('v', 3, 1)

f = cs.Function('f', [q, v], [liecs.SO3(q).act(v)])
g = f.map(3)
print(g(v1, v2))
#print(np.cross([10, 20, 30], [11, 21, 31]))
