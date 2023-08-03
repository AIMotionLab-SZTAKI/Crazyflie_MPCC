# from sys_model import drone
import numpy as np
import casadi as cs
import liecasadi
from utils import Quaternion, RotationQuaternion, create_vector_from_force

N = 1
nr = 3
nV = 3
nq = 4
nomegaB = 3
ntheta = 1
nvtheta = 0
thetapos = nr + nV + nq + nomegaB
nstates = nr + nV + nq + nomegaB + ntheta + nvtheta

dt = 0.01
f = 1 / dt
n_pathpoints = round(10 * f)
steepness = 1

# drone data
mass = 0.028/1000  # kg
gravitational_accel = 9.80665  # m/s^2

J = cs.diag([1.4e-5, 1.4e-5, 2.17e-5])  # kg*m^2
J_inv = cs.diag([1/1.4e-5, 1/1.4e-5, 1/2.17e-5])

thrustCoefficient = 2.88e-8  # N*s^2
dragCoefficient = 7.24e-10  # N*m*s^2
propdist = 0.092

opti = cs.Opti()

# rn the path is only defined for a radius 1 circular helix
gyokketto = np.sqrt(2)
length_height_factor = np.sqrt(1 + steepness**2)  # d path per dz

state = cs.MX.sym('state', 14, 1)
input = cs.MX.sym('input', 4, 1)
quaternion = cs.MX.sym('quaternion', 4, 1)
vector = cs.MX.sym('vector', 3, 1)
distance = cs.MX.sym('distance', 1, 1)


def put_into_vector(u, location):
    """"put  1x1 value to the end of a cs.MX(location, N) matrix"""
    a = cs.MX.zeros(location-1, N)
    return cs.vertcat(a, u)


def derivative1(x, u):

    r = x[:nr, :]
    V = x[nr:nr+nV, :]
    q = x[nr+nV:nr+nV+nq, :]
    omegaB = x[nr+nV+nq:nr+nV+nq+nomegaB, :]
    vtheta = x[:-1]


    quat_arr = cs.MX.sym('quat_arr', 4, N)
    conj = cs.Function('conj', [quat_arr],
                       [cs.vertcat(-quat_arr[0, :], -quat_arr[1, :], -quat_arr[2, :], quat_arr[3, :])])
    qstar = conj(q)

    rdot = x[nr:nr+nV, :]

    Vdot = RotationQuaternion(q).act(create_vector_from_force(u[0, :]/mass, depth=N)) - create_vector_from_force(np.ones((1, N))*gravitational_accel, depth=N)

    qdot = -0.5 * (Quaternion(omegaB) * Quaternion(q))

    omegaBdot = J_inv @ (u[1:4, :] - cs.cross(x[nr+nV+nq:nr+nV+nq+nomegaB ,:], cs.mtimes(J, x[nr+nV+nq:nr+nV+nq+nomegaB, :])))

    direction = cs.vertcat(-cs.sin(x[thetapos, :] / length_height_factor),
                           cs.cos(x[thetapos, :] / length_height_factor),
                           np.tile(steepness, N).reshape(1, N)) \
                / length_height_factor

    thetadot = cs.sum1(x[nr:nr+nV] * direction)
    return cs.vertcat(rdot, Vdot, qdot, omegaBdot, thetadot)


def derivative2(x, u):
    """
    second version of the state derivative function
    :arg:
        x::cs.MX(14, N):state matrix
        u::cs.MX(4, N):input matrix

    :returns:
        cs.MX(14, N): state derivative matrix
        """


    rdot = x[nr:nr + nV, :]

    V_func = cs.Function('V_func', [quaternion, vector], [liecasadi.SO3(quaternion).act(vector)])
    Vdot = V_func.map(N)(x[nr + nV:nr + nV + nq, :], cs.repmat(cs.DM([[0], [0], [1]]), 1, N)) @ cs.diag(u[0, :] / mass) \
           - create_vector_from_force(np.ones((1, N)) * gravitational_accel, depth=N)

    q_func = cs.Function('q_func', [vector, quaternion], [liecasadi.SO3(quaternion).quaternion_derivative(vector, omega_in_body_fixed=True)])
    qdot = q_func.map(N)(x[nr + nV + nq:nr + nV + nq + nomegaB, :], x[nr + nV:nr + nV + nq, :])

    omegaBdot = J_inv @ (u[1:4, :] - cs.cross(x[nr + nV + nq:nr + nV + nq + nomegaB, :],
                                              cs.mtimes(J, x[nr + nV + nq:nr + nV + nq + nomegaB, :])))

    direction_func = cs.Function('direction_func', [distance], [cs.vertcat(-cs.sin(distance / length_height_factor),
                                                                           cs.cos(distance / length_height_factor),
                                                                           steepness) / length_height_factor])

    thetadot = cs.sum1(x[nr:nr + nV, :] * direction_func.map(N)(x[thetapos, :]))

    return cs.vertcat(rdot, Vdot, qdot, omegaBdot, thetadot)


def dynamics(x, u):
    """ rk4 step for the dynamics, x is the state, u is the rotor rpm input"""
    #rk4 step
    # d2 = cs.Function('d2', [state, input], [derivative2(state, input)])
    # derivative_mapped = d2.map(N)
    k1 = derivative2(x, u)
    k2 = derivative2(x + (k1 * 0.5 * dt), u)
    k3 = derivative2(x + (k2 * 0.5 * dt), u)
    k4 = derivative2(x + k3 * dt, u)

    x_next = x + (k1 + k2 * 2 + k3 * 2 + k4) * dt/6
    quat_norm = cs.diag(cs.power(cs.sum1(cs.power(x_next[nr+nV:nr+nV+nq, :], 2)), 0.5))

    return x_next


def MPCC_sim():
    N = 10
    r0 = np.array([1, 0, 0])
    v0 = np.array([0, gyokketto/2, gyokketto/2])
    q0 = np.array([-0.3826834324, 0, 0, 0.9238795325])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.vertcat(r0, v0, q0, omegaB0, 0)

    opti = cs.Opti()
    X = opti.variable(nstates, N+1)
    U = opti.variable(4, N)

    path_direction = cs.MX(3, N+1)
    path_direction[0, :] = -cs.sin(X[thetapos, :] / length_height_factor) / length_height_factor
    path_direction[1, :] = cs.cos(X[thetapos, :]) / length_height_factor
    path_direction[2, :] = steepness / length_height_factor

    path_position = cs.MX(3, N+1)
    path_position[0, :] = cs.cos(X[thetapos, :] / length_height_factor)
    path_position[1, :] = cs.sin(X[thetapos, :])
    path_position[2, :] = steepness * X[thetapos, :] / length_height_factor

    # dynamics constraints
    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[:, 1:] == dynamics(X[:, 0:-1], U))

    # input constraint
    #opti.subject_to(opti.bounded(0, cs.vec(U[0, :])), 3)
    opti.subject_to(opti.bounded(-1e-2, cs.vec(U), 1e-2))

    # cost function weights
    qc = cs.DM.ones((1, N))
    ql = cs.DM_eye(N)
    nu = cs.DM.ones(N)

    # error vector
    err = X[0:nr, 1:] - path_position[:, 1:]

    # lag error vector
    err_lag = cs.sum1(err * path_direction[:, 1:])

    # contouring error vector
    err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

    # path speed
    vtheta = cs.sum1(X[nr:nr+nV, 1:] * path_direction[:, 1:])

    # cost function
    opti.minimize(err_lag @ ql @ err_lag.T + cs.dot(cs.sum1(err_con ** 2), qc) - vtheta @ nu)

    ulist = np.array([[], [], [], []])
    poslist = np.array([[], [], []])
    for i in range(50):

        # calling the solver
        p_opts = {"expand": False}
        s_opts = {"max_iter": 100}
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()
        # print('x:', np.array2string(sol.value(X[:, :]), separator=", "))
        # print('u:', np.array2string(sol.value(U[:, :]), separator=", "))
        N = 1
        x0 = dynamics(x0, sol.value(U)[:, 0, None])
        N = 10
        # set initial X guess
        X_initial = np.array(sol.value(X[:, 2:]))
        X_initial = np.append(X_initial, X_initial[:, -1, None], axis=1)
        opti.set_initial(X[:, 1:], cs.DM(X_initial))

        # set initial U guess
        U_initial = np.array(sol.value(U[:, 1:]))
        U_initial = np.append(U_initial, U_initial[:, -1, None], axis=1)
        opti.set_initial(U, U_initial)
        ulist = np.append(ulist, U_initial[:, 0, None], axis=1)
        poslist = np.append(poslist, x0[0:3, 0], axis=1)
        X[:, 0] = x0

        # print("X_initial:", X_initial)
        # print("U_initial:", U_initial)
    return ulist, poslist

    """one line rk4 step:
    k1 = derivative(x, u, path_direction)
    k2 = derivative(x + (derivative(x, u, path_direction) * 0.5), u, path_direction)
    k3 = derivative(x + (derivative(x + (derivative(x, u, path_direction) * 0.5), u, path_direction) * 0.5), u, path_direction)
    k4 = derivative(x + derivative(x + (derivative(x + (derivative(x, u, path_direction) * 0.5), u, path_direction) * 0.5), u, path_direction), u, path_direction))
    x + (derivative(x, u, path_direction)
     + derivative(x + (derivative(x, u, path_direction) * 0.5), u, path_direction) * 2
     + derivative(x + (derivative(x + (derivative(x, u, path_direction) * 0.5), u, path_direction) * 0.5), u, path_direction) * 2
      + derivative(x + derivative(x + (derivative(x + (derivative(x, u, path_direction) * 0.5), u, path_direction) * 0.5), u, path_direction), u, path_direction))) * dt/6
    dynamics = cs.Function('dynamics', [X, u, theta], [X + (derivative(X, u, path_direction)
                                                            + derivative(X + (derivative(X, u, path_direction) * dt * 0.5), u, path_direction) * 2
                                                            + derivative(X + (derivative(X + (derivative(X, u, path_direction) * dt * 0.5), u, path_direction) * dt * 0.5), u, path_direction) * 2
                                                            + derivative(X + derivative(X + (derivative(X + (derivative(X, u, path_direction) * dt * 0.5), u, path_direction) * dt * 0.5), u, path_direction), u, path_direction)) * dt / 6])"""


