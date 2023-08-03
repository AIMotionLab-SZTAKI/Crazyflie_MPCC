#from sys_model import drone
import casadi
import numpy as np
import casadi as cs
import liecasadi
import matplotlib.pyplot as plt


N = 10
nr = 3
nV = 3
nq = 4
nomegaB = 3
ntheta = 1
nstates = nr + nV + nq + nomegaB + ntheta

dt = 0.05
f = 1 / dt
n_pathpoints = round(10 * f)
steepness = 1

# drone data
mass = 0.028 / 1000  # kg
gravitational_accel = 9.80665  # m/s^2

J = np.array([[1.4e-5, 0, 0], [0, 1.4e-5, 0], [0, 0, 2.17e-5]])  # kg*m^2
J_inv = np.linalg.inv(J)

thrustCoefficient = 2.88e-8  # N*s^2
dragCoefficient = 7.24e-10  # N*m*s^2
propdist = 0.092

# input conversion matrix is used to convert from rotor rpms to forces and torques
a = np.array([[1], [propdist/np.sqrt(2)], [propdist/np.sqrt(2)], [dragCoefficient/thrustCoefficient]])
b = np.array([[1, 1, 1,1 ], [-1, -1, 1, 1], [1, -1, -1, 1], [1, -1, 1, -1]])
input_conversion_matrix = a*b

opti = cs.Opti()

# rn the path is only defined for the a radius 1 circular helix with arbitrary steepness
gyokketto = np.sqrt(2)
length_height_factor = np.sqrt(1 + steepness**2)


def qqmult(q1, q2) -> cs.MX:
    """multiply two quaternions"""
    M = q1.shape[1]
    a = liecasadi.SO3(q1[:, 0])
    b = liecasadi.SO3(q2[:, 0])
    q3 = (a*b).xyzw
    for i in range(1, M) :
        a = liecasadi.SO3(q1[:, i])
        b = liecasadi.SO3(q2[:, i])
        q3 = cs.horzcat(q3, (a*b).xyzw)
    return q3

def qvmult(q1, q2)->cs.MX:
    """multiply a quaternion and a vector"""
    M = q1.shape[1]
    q2 = cs.vertcat(q2, type(q2)(1, M))
    a = liecasadi.SO3(q1[:, 0])
    b = liecasadi.SO3(q2[:, 0])
    q3 = (a*b).xyzw
    for i in range(1, M) :
        a = liecasadi.SO3(q1[:, i])
        b = liecasadi.SO3(q2[:, i])
        q3 = cs.horzcat(q3, (a*b).xyzw)
    return q3

def vqmult(q1, q2)->cs.MX:
    """multiply a vector and a quaternion"""
    M = q2.shape[1]
    q1 = cs.vertcat(q1, type(q1)(1, M))
    a = liecasadi.SO3(q1[:, 0])
    b = liecasadi.SO3(q2[:, 0])
    q3 = (a*b).xyzw
    for i in range(1, M) :
        a = liecasadi.SO3(q1[:, i])
        b = liecasadi.SO3(q2[:, i])
        q3 = cs.horzcat(q3, (a*b).xyzw)
    return q3


def put_into_vector(u, location):
    """"put  1x1 value to the end of a cs.MX(location, N) matrix"""
    a = cs.MX.zeros(location-1, N)
    return cs.vertcat(a, u)


def derivative(x, u):
    """state derivative for the rk4 step,
    returns a vector the same shape as """
    u = cs.mtimes(input_conversion_matrix, thrustCoefficient * (u ** 2))

    # taking the state apart to state variables (maybe this is not necessary)
    V = x[nr:nr+nV, :]
    q = x[nr+nV:nr+nV+nq, :]
    quat_arr = cs.MX.sym('quat_arr', 4, N)
    conj = cs.Function('conj', [quat_arr], [cs.vertcat(-quat_arr[0, :], -quat_arr[1, :], -quat_arr[2, :], quat_arr[3, :])])
    qstar = conj(q)
    omegaB = x[nr+nV+nq:-1, :]
    theta = x[-1:, :]

    # gettting the state derivatives
    rdot = V


    vdot = qqmult(qvmult(q, put_into_vector(u[0, :], 3)), qstar)[1:, :] / mass \
           - cs.repmat(cs.vertcat(0, 0, gravitational_accel), 1, N)

    qdot = qvmult(q, omegaB) * 0.5

    omegaBdot = J_inv @ (u[1:4, :] - cs.cross(x[nr+nV+nq:-1, :], cs.mtimes(J, x[nr+nV+nq:-1, :])))

    # direction of the path, with unit length
    direction = cs.vertcat(-cs.sin(theta/length_height_factor),
                           cs.cos(theta/length_height_factor),
                           np.tile(steepness, N).reshape(1, N)) \
                / length_height_factor
    print(direction)
    # this is just a dot product
    thetadot = cs.sum1(V * direction)
    print("F")
    # the concatenation of the state derivatives to get one state vectot again
    return cs.vertcat(rdot, vdot, qdot, omegaBdot, thetadot)


def dynamics(x, u):
    """ rk4 step for the dynamics, x is the state, u is the rotor rpm input"""

    #rk4 step
    k1 = derivative(x, u)
    k2 = derivative(x + (k1 * 0.5 * dt), u)
    k3 = derivative(x + (k2 * 0.5 * dt), u)
    k4 = derivative(x + k3 * dt, u)

    # maybe i should normalize the quaternions here
    return x + (k1 + k2 * 2 + k3 * 2 + k4) * dt/6

def prediction(x0):
    """run one optimization, later this will be put into the main function
    N+1 states and N inputs, resulting in a final state, constraints and cost function are only applied to the last N states"""

    X = opti.variable(nstates, N+1)

    # taking the state apart to state variables (they are not useful since the no all states are used in the cost function)
    r = X[0:nr, :]
    V = X[nr:nr+nV, :]
    q = X[nr+nV:nr+nV+nq, :]
    omegaB = X[nr+nV+nq:nr+nV+nq+nomegaB, :]
    theta = X[-1:, :]

    U = opti.variable(4, N)
    X[:, 0] = x0
    # dynamics constraints
    opti.subject_to(X[:, 1:] == dynamics(X[:, 0:-1], U))

    # input constraints
    opti.subject_to(opti.bounded(0, cs.vec(U), 1000))

    # path parametrization, only the last N states are considered in the cost function
    #theta_hat_k is the desired position on the path
    # pi_k is tangent to the path, length one
    theta_hat_k = cs.vertcat(cs.cos(X[-1, 1:]/length_height_factor),
                             cs.sin(X[-1, 1:]/length_height_factor),
                             X[-1, 1:]/length_height_factor)

    pi_k = cs.vertcat(-cs.sin(X[-1, 1:]/length_height_factor),
                      cs.cos(X[-1, 1:]/length_height_factor),
                      np.tile(steepness, (1, N))) \
           / length_height_factor

    # cost function, this is technically a manhattan distance squared, so probably there is a smarter way to do it
    err = X[0:nr, 1:] - theta_hat_k

    # contouring error: (1xN), square of the magnitude of the error projected to the normal plane
    # err_con = cs.sum2(cs.power(pi_k @ (cs.sum1(err*pi_k) @ cs.DM_eye(N)).T, 2))
    #alternatively:
    err_con = cs.sum1(cs.power((err - pi_k @ cs.diag(cs.sum1(err*pi_k))), 2))

    # contouring error: (1xN), magnitude of error vector projected to the unit tangent vector
    err_lag = cs.sum1(err*pi_k)

    qc = np.ones((1, N))
    ql = casadi.DM_eye(N)
    nu = casadi.GenDM_ones((1, N))
    opti.minimize(cs.dot(qc, err_con) + cs.dot(err_lag, err_lag) - 30*cs.dot(nu, X[-1, 1:]))
    #opti.minimize(cs.dot(qc, err_con) -cs.dot(nu, X[-1, 1:]))

    # calling the solver
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()
    #print(sol.value(X[0, :]))
    #print(sol.value(U[0, :]))
    return sol.value(X), sol.value(U)



if __name__ == '__main__':


    r0 = np.array([1, 0, 0])
    v0 = np.array([0, 0, 0])
    q0 = np.array([1, 0, 0, 0])
    omegaB0 = np.array([0, 0, 0])
    x0 = cs.vertcat(r0, v0, q0, omegaB0, 0)

    p0 = np.array([[0.], [0], [0.]])


    x, u = prediction(x0)

    print("r:", x[:3, :], "\n V:", x[nr:nr+nV, :], "\n q:", x[nr+nV:nr+nV+nq, :], "\n omegaB:", x[nr+nV+nq:-1, :], "\n theta:", x[-1, :])

    print("u:", u)
    """
    x = np.MX_zeros((13, 5))
    x[0, :] = 1
    x[7, :] = 1/np.sqrt(2)
    x[9, :] = 1/np.sqrt(2)
    u = np.ones((4, 5))*100
    print(dynamics(x, u))"""
