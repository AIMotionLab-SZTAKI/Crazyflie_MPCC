import numpy as np
import casadi as cs
import liecasadi as liecs
from MPCC_with_maps import dynamics
from controller import MPCC



J = cs.DM(np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]]))
a = cs.MX.sym('a', 3)
v1 = cs.DM(np.array([[1, 10, 1, 10], [2, 20, 2, 20], [3, 30, 3, 30]])).T
v2 = cs.DM(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
v3 = cs.DM(np.array([1, 2, 3, 4]))
gyokketto = np.sqrt(2)
q = cs.MX.sym('q', 4, 1)
v = cs.MX.sym('v', 3, 1)

f = cs.Function('f', [q, v], [liecs.SO3(q).act(v)])
g = f.map(3)

J = cs.diag([1.4e-5, 1.4e-5, 2.17e-5])
J_inv = cs.diag([1/1.4e-5, 1/1.4e-5, 1/2.17e-5])
3.999999381155246
0.8311517423459485
-0.9854816256826338
-1.1981102334839298

u = MPCC.torque_scaling(cs.DM([[4], [0.8311517423459485], [-0.9854816256826338], [-1.1981102334839298]]))
omega = cs.DM([[0],
  [0],
  [0]])
def der(tau, omega):
    omegaBdot = J_inv @ (tau[1:] - cs.cross(omega, cs.mtimes(J, omega)))
    return omegaBdot

r0 = np.array([1, 0, 0])
v0 = np.array([0, gyokketto/2, gyokketto/2])
q0 = np.array([0, 0, 0, 1])
omegaB0 = np.array([0, 0, 0])
x0 = cs.DM(cs.vertcat(r0, v0, q0, omegaB0, 0))

x0_mat = cs.repmat(x0, 1, 1)
u_mat = cs.repmat(u, 1, 1)
control = dynamics(x0_mat, u_mat)
"""
0.99999 99988171033
0.014142161423363634
0.012180830463263638
-2.415311683713044e-07
0.7071093858427471
0.5109762193256931
-0.42804520011692027
-0.052035305745572136
0.1295033372818031
0.8928875460464717
-10.080058483045118
-19.99992531945695
19.998622113506
0.01861286621777216"""

k1 = der(u, omega)
k2 = der(u, omega + k1 * 0.02 / 2)
k3 = der(u, omega + k2 * 0.02 / 2)
k4 = der(u, omega + k3 * 0.02)
print(k1, k2, k3, k4)
print((k1 + k2 * 2 + k3 * 2 + k4) * 0.02 / 6)
print(control)
#print(np.cross([10, 20, 30], [11, 21, 31]))
