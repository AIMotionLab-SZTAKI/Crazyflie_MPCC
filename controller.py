import casadi as cs
import numpy as np
import liecasadi
from utils import Quaternion, RotationQuaternion, create_vector_from_force
from path import path_direction_func, path_position_func
import time
import matplotlib.pyplot as plt

"""class for MPCC controller, model included"""
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
gyokketto = np.sqrt(2)

class MPCC:

    def __init__(self, horizon, dt=0.02):
        self.horizon = horizon
        self.dt = dt
        horizon = horizon
        # opti initialization
        self.opti = cs.Opti()

        # decision variables
        self.X = self.opti.variable(n_states, horizon + 1)
        self.U = self.opti.variable(4, horizon)
        self.U_scaled = self.torque_scaling(self.U)
        self.V = self.opti.variable(1, horizon)

        # parameters
        self.X0 = self.opti.parameter(n_states, 1)

        # cost function
        self.opti.minimize(self._cost_function())

        # dynamics constraints
        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.subject_to(self.X[:, 1:] == self.dynamics_mulitple(self.X[:, 0:-1], self.U_scaled, self.V))
        self.opti.subject_to(self.opti.bounded(-20, cs.vec(self.X[10:13, :]), 20))

        # input constraints
        self.opti.subject_to(self.opti.bounded(0, cs.vec(self.U[0, :]), 4))
        self.opti.subject_to(self.opti.bounded(-5, cs.vec(self.U[1:, :]), 5))

        # solver
        self._initalize_solver()

    def _cost_function(self) -> None:
        # cost function weights
        qc = 0.5 * cs.DM.ones((1, self.horizon))
        ql = 2 * cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon).T

        path_position = path_position_func.map(self.horizon+1)(self.X[pos_theta, :])

        path_direction = path_direction_func.map(self.horizon+1)(self.X[pos_theta, :])

        # error vector
        err = self.X[0:pos_V, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        # cost function
        return err_lag @ ql @ err_lag.T + cs.dot(cs.sum1(err_con ** 2), qc) - cs.dot(self.V, nu)

    def _initalize_solver(self, max_iter=3000) -> None:
        """ initializes the solver, max_iter and expand are set"""
        p_opts = {"expand": True}
        s_opts = {"max_iter": max_iter, "print_level": 3}
        self.opti.solver("ipopt", p_opts, s_opts)

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
                                         cs.repmat(cs.DM([[0], [0], [1]]), 1, self.horizon)) @ cs.diag(u[0, :] / mass)) \
                    - create_vector_from_force(np.ones((1, self.horizon)) * gravitational_accel, depth=self.horizon)

        q_func = cs.Function('q_func', [vector, quaternion],
                             [liecasadi.SO3(quaternion).quaternion_derivative(vector, omega_in_body_fixed=True)])
        qdot = q_func.map(self.horizon)(x[pos_omegab:pos_theta, :], x[pos_q:pos_omegab, :])

        omegaBdot = J_inv @ (u[1:4, :] - cs.cross(x[pos_omegab:pos_theta, :],
                                                  cs.mtimes(J, x[pos_omegab:pos_theta, :])))
        placeholder = cs.DM.zeros((1, self.horizon))
        return cs.vertcat(rdot, Vdot, qdot, omegaBdot, placeholder)

    def dynamics_mulitple(self, x: cs.MX, u: cs.MX, v:cs.MX) -> cs.MX:
        """ rk4 step for the dynamics, x is the state, u is the rotor rpm input"""
        k1 = self.derivative_multiple(x, u)
        k2 = self.derivative_multiple(x + (k1 * 0.5 * self.dt), u)
        k3 = self.derivative_multiple(x + (k2 * 0.5 * self.dt), u)
        k4 = self.derivative_multiple(x + k3 * self.dt, u)

        x_next = x + (k1 + k2 * 2 + k3 * 2 + k4) * self.dt / 6
        x_next[pos_theta, :] = x[pos_theta, :] + v * self.dt

        return x_next

    def optimization_step(self, x0, prev_X=None, prev_U=None, prev_V=None, U_is_initial=False) -> (np.ndarray, np.ndarray):
        """
            performs one optimization step, with
        :arg:
            x0::np.ndarray(14, 1): initial state
            prev_X::np.ndarray(14, N+1): previous state trajectory
            prev_U::np.ndarray(4, N): previous input trajectory
        :returns:
            x_traj_planned::np.ndarray(14, N+1): planned state trajectory
            u_traj_planned::np.ndarray(4, N): planned input trajectory
            """

        # set U initial guess
        if prev_U is not None:
            if U_is_initial:
                self.opti.set_initial(self.U, prev_U)
            else:
                U_initial = np.array(prev_U[:, 1:])
                U_initial = np.append(U_initial, U_initial[:, -1, None], axis=1)
                self.opti.set_initial(self.U, U_initial)

        # set X initial guess
        if prev_X is not None:
            X_initial = np.array(prev_X[:, 2:])
            X_initial = np.append(X_initial, X_initial[:, -1, None], axis=1)
            self.opti.set_initial(self.X[:, 1:], X_initial)
        self.opti.set_initial(self.X[:, 0], x0)
        self.opti.set_value(self.X0, x0)

        # set intial guess
        if prev_V is not None:
            V_initial = np.array(prev_V[1:])
            V_initial = np.append(V_initial, V_initial[-1])
            self.opti.set_initial(self.V, V_initial)

        # solve
        sol = self.opti.solve()
        return sol

    def generate_full_trajectory(self, step_no, x0=None, x_intial=None, u_initial=None, return_x=False) -> (np.ndarray, np.ndarray):
        """
        generates a full trajectory with the given initial conditions
        :arg:
            step_no::int: trajectory length in timesteps
            x_intial::np.ndarray(14, 1): initial state
            u_initial::np.ndarray(4, 1): initial input
        :returns:
            x_traj_planned::np.ndarray(14, N+1): planned state trajectory
            u_traj_planned::np.ndarray(4, N): planned input trajectory
            """
        if x_intial is None:
            x_intial = np.zeros((14, 1))
        if u_initial is None:
            u_initial = np.zeros((4, 1))
        if x0 is None:
            r0 = np.array([2, 0, 0])
            v0 = np.array([0, gyokketto / 2, gyokketto / 2])
            q0 = np.array([0.9238795325, 0, 0, 0.3826834324])
            omegaB0 = np.array([0, 0, 0])
            x0 = cs.vertcat(r0, v0, q0, omegaB0, 0)

        start = time.time()
        timelist = np.array([])

        xlist = np.empty((14, step_no+1))
        ulist = np.empty((4, step_no))
        vlist = np.empty((1, step_no))
        xlist[:, 0:1] = x0

        self.opti.set_value(self.X0, x0)

        sol = self.opti.solve()
        for i in range(step_no):
            x0 = self.dynamics_single(x0, self.torque_scaling(sol.value(self.U)[:, 0, None]), sol.value(self.V)[0], self.dt)
            x0[pos_q:pos_omegab] = Quaternion(x0[pos_q:pos_omegab]).normalized().wxyz

            ulist[:, i:i+1] = self.torque_scaling(sol.value(self.U))[:, 0, None]
            xlist[:, i+1:i+2] = x0
            vlist[:, i:i+1] = sol.value(self.V)[0]

            end = time.time()
            timelist = np.append(timelist, end - start)
            start = time.time()
            # calling the solver
            sol = self.optimization_step(x0, sol.value(self.X), sol.value(self.U), sol.value(self.V), False)
            inf_du = sol.stats()['iterations']['inf_du']
            inf_pr = sol.stats()['iterations']['inf_pr']
            plt.semilogy(np.arange(len(inf_du)), inf_du, label='inf_du')
            plt.semilogy(np.arange(len(inf_pr)), inf_pr, label='inf_pr')
            plt.show()

        if return_x:
            return xlist, ulist, vlist, timelist
        else:
            return ulist, vlist, timelist

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
        thrust_vector = cs.vertcat(cs.DM([[0], [0]]), u[0] / mass)
        Vdot = V_func(x[pos_q:pos_omegab], thrust_vector) - cs.DM([[0], [0], [gravitational_accel]])

        q_func = cs.Function('q_func', [vector, quaternion],
                             [liecasadi.SO3(quaternion).quaternion_derivative(vector, omega_in_body_fixed=True)])
        qdot = q_func(x[pos_omegab:pos_theta], x[pos_q:pos_omegab])

        omegaBdot = J_inv @ (u[1:4] - cs.cross(x[pos_omegab:pos_theta],
                                               cs.mtimes(J, x[pos_omegab:pos_theta])))

        return cs.vertcat(rdot, Vdot, qdot, omegaBdot, 0)

    @classmethod
    def dynamics_single(cls, x, u, v, dt):
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
        x_next[pos_theta] = x[pos_theta] + v * dt

        return x_next

    @staticmethod
    def torque_scaling(u, torque_scaling_xx=25, torque_scaling_yy=25, torque_scaling_zz=25):
        return np.repeat(np.array([[1], [1 / torque_scaling_xx], [1 / torque_scaling_yy], [1 / torque_scaling_zz]]), u.shape[1], axis=1) * u
        # return np.diag([1, 1 / torque_scaling_xx, 1 / torque_scaling_yy, 1 / torque_scaling_zz]) @ u

    @staticmethod
    def V_scaling(v, v_scaling=1):
        return v * v_scaling
