import casadi as cs
import numpy as np
import liecasadi
from utils import Quaternion, RotationQuaternion, create_vector_from_force
from path import path_direction_func, path_position_func, linearize_path
import time
import matplotlib.pyplot as plt

"""class for MPCC controller, model included"""
state_non_incrementable = cs.MX.sym('state', 10, 1)  # non incrementable part of the state vector
input = cs.MX.sym('input', 4, 1)
quaternion = cs.MX.sym('quaternion', 4, 1)
vector = cs.MX.sym('vector', 3, 1)
scalar = cs.MX.sym('scalar', 1, 1)

# state variable dimensions inside the state vector
sv_dim={'r': 3, 'V': 3, 'q': 4, 'omegab': 3, 'thrust': 1, 'theta': 1, 'vtheta': 1}
n_r = sv_dim['r'] # these are for bakcwards compatibility
n_V = sv_dim['V']
n_q = sv_dim['q']
n_omegab = sv_dim['omegab']
n_thrust = sv_dim['thrust']
n_theta = sv_dim['theta']
n_vtheta = sv_dim['vtheta']

# state variable positions in the state vector
sv_pos = {'r': 0,
          'V': n_r,
          'q': n_r+n_V,
          'omegab': n_r+n_V+n_q,
          'thrust': n_r+n_V+n_q+n_omegab,
          'theta': n_r+n_V+n_q+n_omegab+n_thrust,
          'vtheta': n_r+n_V+n_q+n_omegab+n_thrust+n_theta}
pos_r = 0
pos_V = sv_pos['V']
pos_q = sv_pos['q']
pos_omegab = sv_pos['omegab']
pos_thrust = sv_pos['thrust']
sv_pos['theta'] = sv_pos['theta']
pos_vtheta = sv_pos['vtheta']

n_states = sum(sv_dim.values())  # number of states

# drone data
mass = 0.028  # kg
gravitational_accel = 9.80665  # m/s^2

J = cs.diag([1.4e-5, 1.4e-5, 2.17e-5])  # kg*m^2
J_inv = cs.diag([1 / 1.4e-5, 1 / 1.4e-5, 1 / 2.17e-5])

thrustCoefficient = 2.88e-8  # N*s^2
dragCoefficient = 7.24e-10  # N*m*s^2
propdist = 0.092
gyokketto = np.sqrt(2)

columnwise_multiplication = cs.Function('columnwise_multiplication', [scalar, vector], [vector @ scalar])


class MpccIncrementalBRate:

    def __init__(self, horizon, dt=0.02, path=None):
        self.horizon = horizon
        self.dt = dt
        horizon = horizon

        # opti initialization
        self.opti = cs.Opti()

        # decision variables
        self.X = self.opti.variable(n_states, horizon + 1)  # state vector
        self.U = self.opti.variable(5, horizon)  # thrust, body rate and path velocity increments

        # parameters for path linearization
        self.X0 = self.opti.parameter(n_states, 1)
        self.line_start_location = self.opti.parameter(3, 1)
        self.line_dir = self.opti.parameter(3, 1)
        self.line_start_along_path = self.opti.parameter(1, 1)


        if path is not None:
            self.path_obj = path
            err_lag, err_con, vel = self._cost_function_spline()
        else:
            err_lag, err_con, vel = self._cost_function_old()
        # cost function
        self.opti.minimize(err_lag + err_con - vel)

        # dynamics constraints
        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.subject_to(self.X[:, 1:] == self.dynamics_mulitple(self.X[:, 0:-1], self.U))

        self.opti.subject_to(0 <= self.X[13, 1:])
        # self.opti.subject_to(self.opti.bounded(-18, cs.vec(self.X[10:13, ]), 18))

        # input constraints
        self.opti.subject_to(self.opti.bounded(0, cs.vec(self.U[0, :]), 400))
        self.opti.subject_to(self.opti.bounded(-800, cs.vec(self.U[1:4, :]), 800))
        self.opti.subject_to(self.opti.bounded(-100, cs.vec(self.U[4, :]), 100))
        self.opti.subject_to(self.opti.bounded(-10, cs.vec(self.X[sv_pos['vtheta'], :]), 400))

        # reality contraints
        self.opti.subject_to(self.opti.bounded(-20, cs.vec(self.X[pos_V:pos_q, 2:]), 20))
        self.opti.subject_to(self.opti.bounded(-20, cs.vec(self.X[10:13, 2:]), 20))
        self.opti.subject_to(self.opti.bounded(0, cs.vec(self.X[sv_pos['thrust'], 2:]), 10))
        # solver
        self._initalize_solver()

    def _cost_function_old(self) -> tuple:
        # cost function weights
        qc = 2 * cs.DM.ones((1, self.horizon))
        ql = 0.4 * cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon).T

        path_position = path_position_func.map(self.horizon+1)(self.X[sv_pos['theta'], :])

        path_direction = path_direction_func.map(self.horizon+1)(self.X[sv_pos['theta'], :])

        # error vector
        err = self.X[0:pos_V, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        return err_lag @ ql @ err_lag.T, cs.dot(cs.sum1(err_con ** 2), qc), cs.dot(self.X[sv_pos['vtheta'], 1:], nu)

    def error(self, position, path_progress):
        "calculate the error for one point"
        err = position - self.line_start_location + self.line_dir @ (path_progress - self.line_start_along_path)
        err_lag = cs.dot(err, self.line_dir)
        err_con = err - self.line_dir * err_lag
        return err_lag, err_con

    def _cost_function_linear(self) -> tuple:
        """ cost function for path linearization"""
        # cost function weights
        qc = 2 * cs.DM.ones((1, self.horizon))
        ql = 0.4 * cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon).T
        qc[:4] = qc[:4] * 2
        path_position = cs.repmat(self.line_start_location, 1, self.horizon+1) + cs.repmat((self.X[sv_pos['theta'], :] - cs.repmat(self.line_start_along_path, 1, self.horizon+1)), 3, 1) * cs.repmat(self.line_dir, 1, self.horizon+1)

        path_direction = cs.repmat(self.line_dir, 1, self.horizon+1)

        # error vector
        err = self.X[0:pos_V, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        return err_lag @ ql @ err_lag.T, cs.dot(cs.sum1(err_con ** 2), qc), cs.dot(self.X[sv_pos['vtheta'], 1:], nu)

    def _cost_function_spline(self):

        # cost function weights
        qc = 2 * cs.DM.ones((1, self.horizon))
        ql = 0.4 * cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon).T

        pp_func = cs.Function('pp_func', [scalar], [self.path_obj.get_path_parameters(scalar)[0]])
        pd_func = cs.Function('pd_func', [scalar], [self.path_obj.get_path_parameters(scalar)[1]])

        path_position = pp_func.map(self.horizon + 1)(self.X[sv_pos['theta'], :])

        path_direction = pd_func.map(self.horizon + 1)(self.X[sv_pos['theta'], :])

        # error vector
        err = self.X[0:pos_V, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        return err_lag @ ql @ err_lag.T, cs.dot(cs.sum1(err_con ** 2), qc), cs.dot(self.X[sv_pos['vtheta'], 1:], nu)

    def _initalize_solver(self, max_iter=1000) -> None:
        """ initializes the solver, max_iter and expand are set"""
        p_opts = {"expand": False}
        s_opts = {"max_iter": max_iter, "print_level": 5, 'fast_step_computation': "no", 'acceptable_tol': 1e-6, 'nlp_scaling_max_gradient': 10, 'nlp_scaling_min_value': 1e-2, 'acceptable_iter': 5, 'linear_solver': 'mumps', 'alpha_for_y': 'max', 'max_iter': max_iter}
        self.opti.solver("ipopt", p_opts, s_opts)

    def derivative_multiple(self, x: cs.MX) -> cs.MX:
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
                                         cs.repmat(cs.DM([[0], [0], [1]]), 1, self.horizon)) @ cs.diag(x[sv_pos['thrust'], :] / mass)) \
                    - create_vector_from_force(np.ones((1, self.horizon)) * gravitational_accel, depth=self.horizon)

        q_func = cs.Function('q_func', [vector, quaternion],
                             [liecasadi.SO3(quaternion).quaternion_derivative(vector, omega_in_body_fixed=True)])
        qdot = q_func.map(self.horizon)(x[sv_pos['omegab']:sv_pos['thrust'], :], x[sv_pos['q']:sv_pos['omegab'], :])

        placeholder = cs.DM.zeros((6, self.horizon))

        return cs.vertcat(rdot, Vdot, qdot, placeholder)

    def dynamics_mulitple(self, x: cs.MX, u: cs.MX) -> cs.MX:
        """ rk4 step for the dynamics, x is the state, u is the rotor rpm input"""
        k1 = self.derivative_multiple(x)
        k2 = self.derivative_multiple(x + (k1 * 0.5 * self.dt))
        k3 = self.derivative_multiple(x + (k2 * 0.5 * self.dt))
        k4 = self.derivative_multiple(x + k3 * self.dt)

        x_next = x + (k1 + k2 * 2 + k3 * 2 + k4) * self.dt / 6
        x_next[sv_pos['omegab']:sv_pos['thrust'], :] = x[sv_pos['omegab']:sv_pos['thrust'], :] + u[1:4, :] * self.dt
        x_next[sv_pos['thrust'], :] = x[sv_pos['thrust'], :] + u[0, :] * self.dt
        x_next[sv_pos['theta'], :] = x[sv_pos['theta'], :] + x[sv_pos['vtheta'], :] * self.dt
        x_next[sv_pos['vtheta'], :] = x[sv_pos['vtheta'], :] + u[4, :] * self.dt
        return x_next

    def optimization_step(self, x0, prev_X=None, prev_U=None, U_is_initial=False) -> (np.ndarray, np.ndarray):
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

        # solve
        sol = self.opti.solve()
        return sol

    def generate_full_trajectory(self, step_no, x0=None, x_intial=None, u_initial=None, return_x=True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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

        ulist = np.empty((5, step_no))
        xlist = [x0]

        self.opti.set_value(self.X0, x0)
        sol = self.opti.solve()

        # update state
        x0 = self.ode_dynamics(x0, sol.value(self.U)[:, 0, None], self.dt)
        x0[pos_q:pos_omegab] = Quaternion(x0[pos_q:pos_omegab]).normalized().wxyz

        # log variables
        ulist[:, 0:1] = sol.value(self.U)[:, 0, None]
        xlist.append(x0)

        # linearize path
        line_start_location, line_dir = linearize_path(x0[sv_pos['theta']])
        self.opti.set_value(self.line_start_location, line_start_location)
        self.opti.set_value(self.line_start_along_path, x0[sv_pos['theta']])
        self.opti.set_value(self.line_dir, line_dir)

        # err_lag, err_con, vel = self._cost_function_new()
        # self.opti.minimize(err_lag + err_con - vel)

        end = time.time()
        timelist = np.append(timelist, end - start)
        for i in range(1, step_no):
            # calling the solver
            sol = self.optimization_step(x0, sol.value(self.X), sol.value(self.U), U_is_initial=False)

            # self.graph_infeasiblities(sol)

            # update position
            x0 = self.ode_dynamics(x0, sol.value(self.U)[:, 0, None], self.dt)
            x0[pos_q:pos_omegab] = Quaternion(x0[pos_q:pos_omegab]).normalized().wxyz

            # log variables
            ulist[:, i:i + 1] = sol.value(self.U)[:, 0, None]
            xlist.append(x0)

            timelist = np.append(timelist, time.time() - end)
            end = time.time()



        if return_x:
            return xlist, ulist, timelist
        else:
            return ulist, timelist

    def graph_infeasiblities(self, sol):
        inf_du = sol.stats()['iterations']['inf_du']
        inf_pr = sol.stats()['iterations']['inf_pr']
        plt.semilogy(np.arange(len(inf_du)), inf_du, label='inf_du')
        plt.semilogy(np.arange(len(inf_pr)), inf_pr, label='inf_pr')
        plt.show()

    @classmethod
    def derivative_single(cls, x, u):
        """
            !!!!numpy arrays are not accepted as input!!!!
            state derivative function for a single state, only for the ode parts of the dynamics are included
            for usage in the ode solver
            :arg:
                x::cs.MX(10, 1):state matrix
                u::cs.MX(4, 1):input matrix, consitsting of body rate and thrust

            :returns:
                cs.MX(14, 1): state derivative matrix
                """
        rdot = x[pos_V:pos_q]

        V_func = cs.Function('V_func', [quaternion, vector], [liecasadi.SO3(quaternion).act(vector)])
        thrust_vector = cs.vertcat(cs.DM([[0], [0]]), u[3] / mass)
        Vdot = V_func(x[pos_q:pos_omegab], thrust_vector) - cs.DM([[0], [0], [gravitational_accel]])

        q_func = cs.Function('q_func', [vector, quaternion],
                             [liecasadi.SO3(quaternion).quaternion_derivative(vector, omega_in_body_fixed=True)])
        qdot = q_func(u[0:3], x[sv_pos['q']:sv_pos['omegab']])

        return cs.vertcat(rdot, Vdot, qdot)

    @staticmethod
    def torque_scaling(u, torque_scaling_xx=0.02, torque_scaling_yy=0.02, torque_scaling_zz=0.02):
        return np.repeat(np.array([[1], [1 / torque_scaling_xx], [1 / torque_scaling_yy], [1 / torque_scaling_zz]]), u.shape[1], axis=1) * u
        # return np.diag([1, 1 / torque_scaling_xx, 1 / torque_scaling_yy, 1 / torque_scaling_zz]) @ u

    @staticmethod
    def V_scaling(v, v_scaling=1):
        return v * v_scaling

    @classmethod
    def ode_dynamics(cls, x0, u0, dt):
        ode = {'x': state_non_incrementable, 'u': input, 'ode': cls.derivative_single(state_non_incrementable, input)}
        F = cs.integrator('F', 'cvodes', ode, {'t0': 0, 'tf': dt})
        res = F(x0=x0[:sv_pos['omegab']], u=x0[sv_pos['omegab']:sv_pos['theta']])['xf']

        res = cs.vertcat(res, 0, 0, 0, 0, 0, 0)
        res[sv_pos['omegab']:sv_pos['thrust']] = res[sv_pos['omegab']:sv_pos['thrust']] + u0[1:4] * dt
        res[sv_pos['thrust']] = res[sv_pos['thrust']] + u0[0] * dt
        res[sv_pos['theta']] = res[sv_pos['theta']] + x0[sv_pos['vtheta']] * dt
        res[sv_pos['vtheta']] = res[sv_pos['vtheta']] + u0[4] * dt

        return res


