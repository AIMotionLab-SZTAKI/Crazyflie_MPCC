import casadi as cs
import numpy as np
import liecasadi
from utils import Quaternion, RotationQuaternion, create_vector_from_force
from path import path_direction_func, path_postion_func

state = cs.MX.sym('state', 14, 1)
input = cs.MX.sym('input', 4, 1)
quaternion = cs.MX.sym('quaternion', 4, 1)
vector = cs.MX.sym('vector', 3, 1)

# state variable dimensions
n_r = 3
n_V = 3
n_q = 4
n_omegab = 3
n_theta = 1

# state variable positions in the state vector
pos_r = 0
pos_V = n_r
pos_q = pos_V + n_V
pos_omegab = pos_q + n_q
pos_theta = pos_omegab + n_omegab
n_states = n_r + n_V + n_q + n_omegab + n_theta

# drone data
mass = 0.028  # kg
gravitational_accel = 9.80665  # m/s^2

J = cs.diag([1.4e-5, 1.4e-5, 2.17e-5])  # kg*m^2
J_inv = cs.diag([1 / 1.4e-5, 1 / 1.4e-5, 1 / 2.17e-5])

thrustCoefficient = 2.88e-8  # N*s^2
dragCoefficient = 7.24e-10  # N*m*s^2
propdist = 0.092


class MPCC:

    def __init__(self, horizon, x0, dt=0.02):
        self.horizon = horizon
        self.dt = dt

        # opti initialization
        self.opti = cs.Opti()

        # decision variables
        self.X = self.opti.variable(n_states, horizon + 1)
        self.U = self.opti.variable(4, horizon)

        self.U_scaled = self.torque_scaling(self.U)

        # dynamics constraints
        self.opti.subject_to(self.X[:, 0] == x0)
        self.opti.subject_to(self.X[:, 1:] == self.dynamics(self.X[:, 0:-1], self.U_scaled))
        self.opti.subject_to(self.opti.bounded(-20, cs.vec(self.X[10:13, :]), 20))

        # input constraints
        self.opti.subject_to(self.opti.bounded(0, cs.vec(self.U[0, :]), 4))
        self.opti.subject_to(self.opti.bounded(-6, cs.vec(self.U[1:, :]), 6))

        # cost function
        self.initialize_cost_function()

    def initialize_cost_function(self) -> None:
        # cost function weights
        qc = cs.DM.ones((1, self.horizon))
        ql = cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon)

        path_position = path_postion_func.map(self.horizon)(self.X[0:n_r, 1:])

        path_direction = path_direction_func.map(self.horizon)(self.X[0:n_r, 1:])
        # error vector
        err = self.X[0:n_r, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        # path speed
        vtheta = cs.sum1(self.X[pos_V:pos_q, 1:] * path_direction[:, 1:])

        # cost function
        self.opti.minimize(err_lag @ ql @ err_lag.T + cs.dot(cs.sum1(err_con ** 2), qc))

    def derivative_multiple(self, x: cs.MX, u: cs.MX) -> cs.MX:
        """
        second version of the state derivative function
        :arg:
            x::cs.MX(14, N):state matrix
            u::cs.MX(4, N):input matrix

        :returns:
            cs.MX(14, N): state derivative matrix
            """
        rdot = x[pos_V:pos_q, :]

        V_func = cs.Function('V_func', [quaternion, vector], [liecasadi.SO3(quaternion).act(vector)])

        Vdot = (V_func.map(self.horizon)(x[pos_q:pos_omegab, :],
                                         cs.repmat(cs.DM([[0], [0], [1]]), 1, self.horizon)) @ cs.diag(
            u[0, :] / mass)) \
               - create_vector_from_force(np.ones((1, self.horizon)) * gravitational_accel, depth=self.horizon)

        q_func = cs.Function('q_func', [vector, quaternion],
                             [liecasadi.SO3(quaternion).quaternion_derivative(vector, omega_in_body_fixed=True)])
        qdot = q_func.map(self.horizon)(x[pos_omegab:pos_theta, :], x[pos_q:pos_omegab, :])

        omegaBdot = J_inv @ (u[1:4, :] - cs.cross(x[pos_omegab:pos_theta, :],
                                                  cs.mtimes(J, x[pos_omegab:pos_theta, :])))

        thetadot = cs.sum1(x[pos_V:pos_q, :] * path_direction_func.map(self.horizon)(x[pos_theta, :]))

        return cs.vertcat(rdot, Vdot, qdot, omegaBdot, thetadot)

    def dynamics_mulitple(self, x, u):
        """ rk4 step for the dynamics, x is the state, u is the rotor rpm input"""
        k1 = self.derivative_multiple(x, u)
        k2 = self.derivative_multiple(x + (k1 * 0.5 * self.dt), u)
        k3 = self.derivative_multiple(x + (k2 * 0.5 * self.dt), u)
        k4 = self.derivative_multiple(x + k3 * self.dt, u)

        x_next = x + (k1 + k2 * 2 + k3 * 2 + k4) * self.dt / 6

        return x_next

    @classmethod
    def derivative_single(cls, x, u):
        """
            !!!!numpy arrays are not accepted as input!!!!
            state derivative function for a single state
            :arg:
                x::cs.MX(14, 1):state matrix
                u::cs.MX(4, 1):input matrix

            :returns:
                cs.MX(14, 1): state derivative matrix
                """
        rdot = x[pos_V:pos_q]

        V_func = cs.Function('V_func', [quaternion, vector], [liecasadi.SO3(quaternion).act(vector)])
        thrust_vector = cs.DM([[0], [0], [u[0] / mass]])
        Vdot = V_func(x[pos_q:pos_omegab, :], thrust_vector) - cs.DM([[0], [0], [gravitational_accel]])

        q_func = cs.Function('q_func', [vector, quaternion],
                             [liecasadi.SO3(quaternion).quaternion_derivative(vector, omega_in_body_fixed=True)])
        qdot = q_func(x[pos_omegab:pos_theta, :], x[pos_q:pos_omegab, :])

        omegaBdot = J_inv @ (u[1:4] - cs.cross(x[pos_omegab:pos_theta, :],
                                               cs.mtimes(J, x[pos_omegab:pos_theta])))

        thetadot = cs.sum1(x[pos_V:pos_q] * path_direction_func(x[pos_theta]))

        return cs.vertcat(rdot, Vdot, qdot, omegaBdot, thetadot)

    @classmethod
    def dynamics_single(cls, x, u, dt):
        """ rk4 integration of the state derivative function
            :arg:
                x::cs.MX(14, 1):state matrix
                u::cs.MX(4, 1):input matrix
                dt::float:time step

            :returns:
                cs.MX(14, 1): state derivative matrix
        """
        # rk4 step
        # d2 = cs.Function('d2', [state, input], [derivative2(state, input)])
        # derivative_mapped = d2.map(N)
        k1 = MPCC.derivative_single(x, u)
        k2 = MPCC.derivative_single(x + (k1 * 0.5 * dt), u)
        k3 = MPCC.derivative_single(x + (k2 * 0.5 * dt), u)
        k4 = MPCC.derivative_single(x + k3 * dt, u)

        x_next = x + (k1 + k2 * 2 + k3 * 2 + k4) * dt / 6

        return x_next

    @staticmethod
    def torque_scaling(u, torque_scaling_xx=1, torque_scaling_yy=1, torque_scaling_zz=1):
        return np.diag([1, 1 / torque_scaling_xx, 1 / torque_scaling_yy, 1 / torque_scaling_zz]) @ u
