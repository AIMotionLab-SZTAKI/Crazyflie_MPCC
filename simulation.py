import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

import controller
#from mpl_toolkits.mplot3d.proj3d import proj_transform
from utils import Quaternion, RotationQuaternion, create_vector_from_force, proj_transform
from trial_mpc import gyokketto
from mpl_toolkits.mplot3d import axes3d
from MPCC_with_maps import dynamics, MPCC_sim
from matplotlib import animation
from path import path_direction_func, path_position_func
from controller import MPCC

"""The 3d arrow drawing is based on WetHats notebook on https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c"""
"""this file only does the plotting of the drone, the model is in MPCC_with_maps.py"""

class Arrow3D(FancyArrowPatch):  # unused

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

    def set_spatial_pos(self, x, y, z, dx, dy, dz):
        self._xyz = x, y, z
        self._dxdydz = dx, dy, dz


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)



setattr(Axes3D, 'arrow3D', _arrow3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t = np.linspace(0, 10, 1000)

body_x_ax = ax.quiver(1, 0, 0, 1, 0, 0, color='r')
body_y_ax = ax.quiver(1, 0, 0, 0, 1, 0, color='g')
body_z_ax = ax.quiver(1, 0, 0, 0, 0, 1, color='b')


def draw_drone(state):
    """Draws a drone at the given state"""
    global body_x_ax
    global body_y_ax
    global body_z_ax
    r = state[0:3]
    qs = state[9]
    qv = state[6:9]
    x = r[0]
    y = r[1]
    z = r[2]
    q = state[6:10]
    q[0] = qs
    q[1:] = qv

    X = RotationQuaternion(q).act(np.array([1, 0, 0]).reshape(3, 1))
    Y = RotationQuaternion(q).act(np.array([0, 1, 0]).reshape(3, 1))
    Z = RotationQuaternion(q).act(np.array([0, 0, 1]).reshape(3, 1))

    body_x_ax.remove()
    body_y_ax.remove()
    body_z_ax.remove()

    body_x_ax = ax.quiver(x, y, z, X[0], X[1], X[2], color='r')
    body_y_ax = ax.quiver(x, y, z, Y[0], Y[1], Y[2], color='g')
    body_z_ax = ax.quiver(x, y, z, Z[0], Z[1], Z[2], color='b')

    path_loc = np.array(path_position_func(state[-1]))
    ax.plot(np.array([x,  path_loc[0, None]])[:, 0, 0], np.array([y, path_loc[1, None]])[:, 0, 0], np.array([z, path_loc[2, None]])[:, 0, 0], color='r')


def draw_bg():
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-4, 6)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.plot(np.cos(t), np.sin(t), t)

ulist = np.array([[ 3.99999938e+00,  1.37357578e-06,  2.52781953e+00,  6.28037186e-01,  1.52488492e-01,  2.81320912e-05,  1.80584113e-04,  3.60755846e-02,  3.18499111e-01,  8.77233819e-05,  1.12297611e-05,  3.96720310e-06,
   2.88868885e-01,  5.20483843e-01,  4.80139399e-01],
 [ 3.48630563e-03, -1.49999096e-02, -6.04124444e-04, -4.88750472e-03,  1.65128755e-04,  1.08505353e-03,  2.03985302e-03,  2.73975603e-03,  3.07996689e-03,  2.48802278e-03,  1.68972343e-03,  1.17613150e-03,
   6.09157256e-04,  1.28188714e-04, -5.87617923e-04],
 [-5.08540543e-03,  1.49999965e-02,  1.77960535e-03, -5.54682820e-04,  1.63939124e-04,  1.06947305e-03,  2.02318068e-03,  2.71702330e-03,  3.07995902e-03, -2.27974257e-03,  1.21149433e-03,  2.98264330e-04,
   7.28106334e-04,  1.07699944e-03, -6.42304891e-03],
 [-7.24216409e-03, -1.44578121e-02,  1.29577820e-02,  7.76022282e-03,  4.28244133e-03,  8.57948906e-03,  4.86647689e-03,  4.95320552e-03,  3.31433076e-07,  9.47516496e-09,  2.76597321e-09,  2.79948893e-10,
   1.45048493e-09, -4.70865697e-09, -1.49999904e-02]])

if __name__=="__main__":
    r0 = np.array([1, 0, 0])
    v0 = np.array([0, gyokketto/2, gyokketto/2])
    q0 = np.array([-0.3826834324, 0, 0, 0.9238795325])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.DM(cs.vertcat(r0, v0, q0, omegaB0, 0))
    sim_length = 15

    # ulist, poslist = MPCC_sim(x0=x0, horizon=20, sim_length=sim_length)
    xlist = [x0]
    # Rotate the axes and update
    for i in range(sim_length):
        xlist.append(dynamics(xlist[-1], ulist[:, i, None]))

    ani = animation.FuncAnimation(fig, draw_drone, frames=xlist, interval=50, repeat=True, init_func=draw_bg())
    print(np.array2string(np.array(ulist), separator=", ", max_line_width=120))
    plt.show()

