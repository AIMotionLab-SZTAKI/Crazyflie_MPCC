"""simulate and display the drone behavior"""
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d.proj3d import proj_transform
from utils import Quaternion, RotationQuaternion, create_vector_from_force, proj_transform
from trial_mpc import gyokketto
from mpl_toolkits.mplot3d import axes3d
from MPCC_with_maps import dynamics, MPCC_sim
from matplotlib import animation

"""The 3d arrow drawing is based on WetHats notebook on https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c"""


class Arrow3D(FancyArrowPatch):# unused

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
t = np.linspace(-5, 5, 1000)

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


def draw_bg():
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-4, 6)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.plot(np.cos(t), np.sin(t), t)

ulist = cs.DM([[ 0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001,
   0.01000001,  0.01000001,  0.01000001,  0.01000001,  0.01000001],
 [-0.00907302, -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ,
  -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 , -0.0090726 ],
 [-0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001,
  -0.01000001, -0.01000001, -0.01000001, -0.01000001, -0.01000001],
 [-0.00999992, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977,
  -0.00999977, -0.00999977, -0.00999977, -0.00999977, -0.00999977]])



if __name__=="__main__":
    r0 = np.array([1, 0, 0])
    v0 = np.array([0, gyokketto/2, gyokketto/2])
    q0 = np.array([-0.3826834324, 0, 0, 0.9238795325])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.DM(cs.vertcat(r0, v0, q0, omegaB0, 0))
    ulist, poslist = MPCC_sim()
    xlist = [x0]
    # Rotate the axes and update
    for i in range(50):
        xlist.append(cs.DM(dynamics(xlist[-1], ulist[:, i, None])))


    ani = animation.FuncAnimation(fig, draw_drone, frames=xlist, interval=100, repeat=True, init_func=draw_bg())
    print(np.array2string(np.array(xlist), separator=", ", max_line_width=120))
    plt.show()

