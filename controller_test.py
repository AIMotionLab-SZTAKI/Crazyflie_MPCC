import controller
from controller import MPCC
import matplotlib.pyplot as plt
import numpy as np
from utils import RotationQuaternion
import casadi as cs
from MPCC_with_maps import dynamics
from matplotlib import animation
import time
from path import path_position_func

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t = np.linspace(-5, 10, 1000)

def test_dynamics():
    """Test the dynamics of the drone"""
    x0 = cs.DM([-0.091, 0.1979, -0.199, -3.235, -4.796, -7.139, -0.39, -0.84, 0.07, -0.34, 20, 20, 9.78, 0])
    u = cs.DM([[4], [9.083e-3], [-1.65160394e-3], [-1.108e-2]])
    inputarray = cs.repmat(u, 1, 4)
    statearray = cs.repmat(x0, 1, 4)
    v = cs.DM.zeros((1, 4))
    states = dynamics(x0, u)
    print(states)
    states2 = controller.MPCC.dynamics_single(x0, u, 0, 0.02)
    mpccer = controller.MPCC(4, dt=0.02)
    states3 =mpccer.dynamics_mulitple(statearray, inputarray, v)
    print(states2)
    print(states3)
    print(x0)
    assert np.allclose(states[:-1], states2[:-1], atol=1e-10)
    assert np.allclose(states[:-1], states3[:-1, :], atol=1e-10)
    assert np.allclose(states2, states3, atol=1e-10)
    assert np.allclose(states3[:, 0], states3[:, 1], atol=1e-10)
    assert np.allclose(states3[:, 1], states3[:, 2], atol=1e-10)
    assert np.allclose(states3[:, 2], states3[:, 3], atol=1e-10)


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
    ax.plot(np.array([x, path_loc[0, None]])[:, 0, 0], np.array([y, path_loc[1, None]])[:, 0, 0],
            np.array([z, path_loc[2, None]])[:, 0, 0], color='r')


def draw_bg():
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-4, 6)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    # ax.plot(np.cos(t), np.sin(t), t)


def test_planning():
    r0 = np.array([1.01, 0, 0])
    v0 = np.array([0, 1, 1])
    q0 = np.array([-0.3826834324, 0, 0, 0.9238795325])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.DM(cs.vertcat(r0, v0, q0, omegaB0, 0))
    sim_length = 25
    dt = 0.02
    MPCC = controller.MPCC(horizon=20, dt=dt)
    u_initial = np.array(
        [[3.99999997e+00, 3.99999996e+00, 3.99999994e+00, 4.00000000e+00, 3.99999989e+00, 3.99999988e+00,
          3.99999988e+00, 3.99999989e+00, 3.99999990e+00, 3.99999991e+00, 3.99999991e+00, 3.99999992e+00,
          3.99999992e+00, 3.99999992e+00, 4.00000000e+00],
         [-4.02489122e-03, 9.85709076e-03, -9.01237654e-03, 1.28874440e-02, -1.15512746e-02, 1.49999753e-02,
          -1.36124096e-02, 1.30913526e-02, -1.45059277e-02, 1.23574380e-02, -1.23313344e-02, 1.10542320e-02,
          -1.13025712e-02, 1.05704141e-02, -1.14031601e-02],
         [-6.95171235e-03, 2.14869715e-03, -5.22094364e-03, 2.03584060e-03, -9.43249730e-03, 1.13489507e-02,
          -9.21215582e-03, 1.49998961e-02, -1.17303816e-02, 1.49998948e-02, -1.41021005e-02, 1.48734657e-02,
          -1.49842128e-02, 1.47719178e-02, -1.49995944e-02],
         [9.74940007e-03, -1.49991579e-02, 1.49998591e-02, -1.49996529e-02, 1.49998966e-02, -1.49997375e-02,
          1.49999334e-02, -1.49999023e-02, 1.49999178e-02, -1.49998731e-02, 1.49998971e-02, -1.49998922e-02,
          1.49998870e-02, -1.49998995e-02, 1.49999583e-02]])

    start = time.time()
    ulist, vlist, timelist = MPCC.generate_full_trajectory(sim_length, x0=x0, u_initial=u_initial)
    end = time.time()
    print('time taken for each iteration:', timelist)
    print('iteration sum:', sum(timelist))
    print('full trajectory generation time:', end-start)
    xlist = [x0]
    # Rotate the axes and update
    for i in range(sim_length):
        xlist.append(controller.MPCC.dynamics_single(xlist[-1], ulist[:, i, None], vlist[:, i, None], dt))

    ax.plot(np.cos(t), np.sin(t), t)
    ani = animation.FuncAnimation(fig, draw_drone, frames=xlist, interval=200, repeat=True, init_func=draw_bg())
    print(np.array2string(np.array(ulist), separator=", ", max_line_width=220))
    print(np.array2string(np.array(xlist), separator=", ", max_line_width=220))
    print(np.array2string(np.array(vlist), separator=", ", max_line_width=220))

    plt.show()

if __name__ == '__main__':
    test_planning()