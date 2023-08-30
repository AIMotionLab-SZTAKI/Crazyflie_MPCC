import controller
import controller_for_bodyrate
import controller_incremental_brate
import matplotlib.pyplot as plt
import numpy as np
from utils import RotationQuaternion
import casadi as cs
from MPCC_with_maps import dynamics
from matplotlib import animation
import time
from path import *

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax_flat = fig.add_subplot(122)
t = np.linspace(0, 50, 100)

p = np.array([[0, 2, 0, 4],
                  [1, 3, 4, 5],
                  [2, 3, 7, 1],
                  [3, 2, 11, 4],
                  [4, 4, 9, 13],
                  [5, 7, 9, 4],
                  [6, 9, 11, 5],
                  [7, 8, 7, 6],
                  [8, 8, 4, 9],
                  [9, 9, 0, 10],
                  [10, 7, 2, 7],
                  [11, 4, 2, 8],
                  [12, 2, 0, 4]])
spline = Spline_3D(p)

def test_dynamics():
    """Test the dynamics of the drone"""
    x0 = cs.DM([-0.091, 0.1979, -0.199, -3.235, -4.796, -7.139, -0.39, -0.84, 0.07, -0.34, 0, 0, 0, 0])
    u = cs.DM([[4], [9.083], [-1.65160394], [-1.108]])
    inputarray = cs.repmat(u, 1, 4)
    statearray = cs.repmat(x0, 1, 4)
    v = cs.DM.zeros((1, 4))
    states = dynamics(x0, u)
    print(states)
    states2 = controller_for_bodyrate.MPCC_for_bodyrate.dynamics_single(x0, u, 0, 0.02)
    mpccer = controller_for_bodyrate.MPCC_for_bodyrate(4, dt=0.02)
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
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_zlim(-1, 12)
    # ax.plot(np.cos(t), np.sin(t), t)


def test_planning_brate():


    r0 = np.array([2, 0, 4])
    v0 = np.array([0, 1, 1])
    q0 = np.array([-0.9238795325, 0, 0, 0.3826834324])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.DM(cs.vertcat(r0, v0, q0, omegaB0, 0))
    sim_length = 44
    dt = 0.01



    MPCC = controller_for_bodyrate.MPCC_for_bodyrate(horizon=10, dt=dt)

    start = time.time()
    xlist, ulist, vlist, timelist = MPCC.generate_full_trajectory(sim_length, x0=x0)
    end = time.time()
    print('time taken for each iteration:', timelist)
    print('iteration sum:', sum(timelist))
    print('full trajectory generation time:', end-start)
    # xlist = [x0]
    # Rotate the axes and update
    # for i in range(sim_length):
        # xlist.append(controller.MPCC.dynamics_single(xlist[-1], ulist[:, i, None], vlist[:, i, None], dt))

    curve = np.array([spline.get_path_parameters(i)[0] for i in t])
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='g')
    ani = animation.FuncAnimation(fig, draw_drone, frames=xlist, interval=200, repeat=True, init_func=draw_bg())
    # print(np.array2string(np.array(ulist), separator=", ", max_line_width=220))
    # print(np.array2string(np.array(xlist), separator=", ", max_line_width=220))
    # print(np.array2string(np.array(vlist), separator=", ", max_line_width=220))

    plt.show()

def test_planning_incremental():
    r0 = np.array([1, 0, 0])
    v0 = np.array([0, 1, 1])
    q0 = np.array([-0.9238795325, 0, 0, 0.3826834324])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.DM(cs.vertcat(r0, v0, q0, 0, 0, 0, 0, 0, 0))
    sim_length = 20
    dt = 0.01

    MPCC = controller_incremental_brate.MpccIncrementalBRate(horizon=10, dt=dt)

    start = time.time()
    xlist, ulist, timelist = MPCC.generate_full_trajectory(sim_length, x0=x0)
    end = time.time()
    print('time taken for each iteration:', timelist)
    print('iteration sum:', sum(timelist))
    print('full trajectory generation time:', end - start)
    # xlist = [x0]
    # for i in range(sim_length):
    # xlist.append(controller.MPCC.dynamics_single(xlist[-1], ulist[:, i, None], vlist[:, i, None], dt))

    curve = np.array([path_position_func(i) for i in t])
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='g')
    ani = animation.FuncAnimation(fig, draw_drone, frames=xlist, interval=200, repeat=True, init_func=draw_bg())
    print(np.array2string(np.array(ulist), separator=", ", max_line_width=220))
    print(np.array2string(np.array(xlist), separator=", "))

    plt.show()


def test_planning_full_nlp():
    r0 = np.array([1.01, 0, 1])
    v0 = np.array([0, 1, -1])
    q0 = np.array([-0.9238795325, 0, 0, 0.3826834324])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.DM(cs.vertcat(r0, v0, q0, omegaB0, 0))
    sim_length = 35
    dt = 0.02

    MPCC = controller.MPCC(horizon=10, dt=dt)


    start = time.time()
    xlist, ulist, vlist, timelist = MPCC.generate_full_trajectory(sim_length, x0=x0)
    end = time.time()
    print('time taken for each iteration:', timelist)
    print('iteration sum:', sum(timelist))
    print('full trajectory generation time:', end-start)
    # xlist = [x0]
    # Rotate the axes and update
    # for i in range(sim_length):
        # xlist.append(controller.MPCC.dynamics_single(xlist[-1], ulist[:, i, None], vlist[:, i, None], dt))

    curve = np.array([path_position_func(i) for i in t])
    ax.plot(curve[:, 0, 0], curve[:, 1, 0], curve[:, 2, 0], color='g')
    ani = animation.FuncAnimation(fig, draw_drone, frames=xlist, interval=200, repeat=True, init_func=draw_bg())
    # print(np.array2string(np.array(ulist), separator=", ", max_line_width=220))
    # print(np.array2string(np.array(xlist), separator=", ", max_line_width=220))
    # print(np.array2string(np.array(vlist), separator=", ", max_line_width=220))

    plt.show()


def test_path():
    draw_bg()

    curve = np.array([path_position_func(i) for i in t])
    ax.plot(curve[:, 0, 0], curve[:, 1, 0], curve[:, 2, 0], color='g')
    theta = np.linspace(0, 2 * 3.14*np.sqrt(2), 8)
    direction = np.array([path_direction_func(i) for i in theta])
    pos = np.array([path_position_func(i) for i in theta])
    # steepness = np.array([steepness_func(i) for i in theta])
    #ax_flat.scatter(theta, steepness[:, 0, 0])
    print(path_position_func(np.sqrt(2)))
    ax.quiver(pos[:, 0, 0], pos[:, 1, 0], pos[:, 2, 0], direction[:, 0, 0], direction[:, 1, 0], direction[:, 2, 0], color='r')
    plt.show()

def test_linarization():
    curve = np.array([path_position_func(i) for i in t])
    ax.plot(curve[:, 0, 0], curve[:, 1, 0], curve[:, 2, 0], color='g')
    for i in range(6):
        point, direction = linearize_path(i)
        line = np.array(point) + direction[:, None] * t[:, None].T
        ax.plot(line[0], line[1], line[2], color='r')
        print(point, direction)
    plt.show()


def test_spline():
    p = np.array([[0, 2, 0, 4],
                  [1, 3, 4, 5],
                  [2, 3, 7, 1],
                  [3, 2, 11, 4],
                  [4, 4, 9, 13],
                  [5, 7, 9, 4],
                  [6, 9, 11, 5],
                  [7, 8, 7, 6],
                  [8, 8, 4, 9],
                  [9, 9, 0, 10],
                  [10, 7, 2, 7],
                  [11, 4, 2, 8],
                  [12, 2, 0, 4]])
    path_spline = Spline_3D(p)
    path_spline.plot_equally()


if __name__ == '__main__':
    test_planning_incremental()
    #test_linarization()