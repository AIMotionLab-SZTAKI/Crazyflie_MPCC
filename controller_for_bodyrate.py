import casadi as cs
import numpy as np
import liecasadi
from utils import Quaternion, RotationQuaternion, create_vector_from_force
from path import path_direction_func, path_position_func, linearize_path
import time
import matplotlib.pyplot as plt

"""class for MPCC controller, model included"""
state = cs.MX.sym('state', 14, 1)
input = cs.MX.sym('input', 4, 1)
quaternion = cs.MX.sym('quaternion', 4, 1)
vector = cs.MX.sym('vector', 3, 1)
scalar = cs.MX.sym('scalar', 1, 1)
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

columnwise_multiplication = cs.Function('columnwise_multiplication', [scalar, vector], [vector @ scalar])

def test_linarization(i):
    t = np.linspace(-1, 2 * 3.14 * np.sqrt(2), 50)
    curve = np.array([path_position_func(i) for i in t])
    plt.plot(curve[:, 0, 0], curve[:, 1, 0], curve[:, 2, 0], color='g')
    point, direction = linearize_path(i)
    line = np.array(point) + np.array(direction) * t
    plt.plot(line[0], line[1], line[2], color='r')
    plt.show()


class MPCC_for_bodyrate:

    def __init__(self, horizon, dt=0.02, path=None):
        self.horizon = horizon
        self.dt = dt
        horizon = horizon
        # opti initialization
        self.opti = cs.Opti()

        # decision variables
        self.X = self.opti.variable(n_states, horizon + 1)  # state vector
        self.U = self.opti.variable(4, horizon)  # thrust and change of bodyrate
        self.U_scaled = self.torque_scaling(self.U)
        self.V = self.opti.variable(1, self.horizon)

        # parameters
        self.X0 = self.opti.parameter(n_states, 1)
        self.line_start_location = self.opti.parameter(3, 1)
        self.line_dir = self.opti.parameter(3, 1)
        self.line_start_along_path = self.opti.parameter(1, 1)


        if path is not None:
            self.path_obj = path
        err_lag, err_con, vel = self._cost_function_spline()
        # cost function
        self.opti.minimize(err_lag + err_con - vel)

        # dynamics constraints
        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.subject_to(self.X[:, 1:] == self.dynamics_mulitple(self.X[:, 0:-1], self.U_scaled, self.V))

        self.opti.subject_to(0 <= self.X[13, 1:])
        # self.opti.subject_to(self.opti.bounded(-18, cs.vec(self.X[10:13, ]), 18))

        # input constraints
        self.opti.subject_to(self.opti.bounded(0, cs.vec(self.U[0, :]), 10))
        self.opti.subject_to(self.opti.bounded(-5/self.dt, cs.vec(self.U[1:, :]), 5/self.dt))
        self.opti.subject_to(self.opti.bounded(-10, cs.vec(self.V), 100))

        # reality contraints
        # self.opti.subject_to(self.opti.bounded(-19, cs.vec(self.X[pos_V:pos_q, 1]), 19))
        # self.opti.subject_to(self.opti.bounded(-20, cs.vec(self.X[pos_V:pos_q, 2:]), 20))
        # self.opti.subject_to(self.opti.bounded(-19, cs.vec(self.X[10:13, 1]), 19))
        self.opti.subject_to(self.opti.bounded(-20, cs.vec(self.X[10:13, 2:]), 20))
        # solver
        self._initalize_solver()

    def _cost_function_old(self) -> tuple:
        # cost function weights
        qc = 2 * cs.DM.ones((1, self.horizon))
        ql = 0.4 * cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon).T

        path_position = path_position_func.map(self.horizon+1)(self.X[pos_theta, :])

        path_direction = path_direction_func.map(self.horizon+1)(self.X[pos_theta, :])

        # error vector
        err = self.X[0:pos_V, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        return err_lag @ ql @ err_lag.T, cs.dot(cs.sum1(err_con ** 2), qc), cs.dot(self.V, nu)

    def error(self, position, path_progress):
        "calculate the error for one point"
        err = position - self.line_start_location + self.line_dir @ (path_progress - self.line_start_along_path)
        err_lag = cs.dot(err, self.line_dir)
        err_con = err - self.line_dir * err_lag
        return err_lag, err_con

    def _cost_function_new(self) -> tuple:

        # cost function weights
        qc = 2 * cs.DM.ones((1, self.horizon))
        ql = 0.4 * cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon).T
        qc[:4] = qc[:4] * 2
        path_position = cs.repmat(self.line_start_location, 1, self.horizon+1) + cs.repmat((self.X[pos_theta ,:] - cs.repmat(self.line_start_along_path, 1, self.horizon+1)), 3, 1) * cs.repmat(self.line_dir, 1, self.horizon+1)

        path_direction = cs.repmat(self.line_dir, 1, self.horizon+1)

        # error vector
        err = self.X[0:pos_V, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        return err_lag @ ql @ err_lag.T, cs.dot(cs.sum1(err_con ** 2), qc), cs.dot(self.V, nu)

    def _cost_function_spline(self):

        # cost function weights
        qc = 2 * cs.DM.ones((1, self.horizon))
        ql = 0.4 * cs.DM_eye(self.horizon)
        nu = 0.001 * cs.DM.ones(self.horizon).T

        pp_func = cs.Function('pp_func', [scalar], [self.path_obj.get_path_parameters(scalar)[0]])
        pd_func = cs.Function('pd_func', [scalar], [self.path_obj.get_path_parameters(scalar)[1]])

        path_position = pp_func.map(self.horizon + 1)(self.X[pos_theta, :])

        path_direction = pd_func.map(self.horizon + 1)(self.X[pos_theta, :])

        # error vector
        err = self.X[0:pos_V, 1:] - path_position[:, 1:]

        # lag error vector
        err_lag = cs.sum1(err * path_direction[:, 1:])

        # contouring error vector
        err_con = err - path_direction[:, 1:] @ cs.diag(cs.sum1(err * path_direction[:, 1:]))

        return err_lag @ ql @ err_lag.T, cs.dot(cs.sum1(err_con ** 2), qc), cs.dot(self.V, nu)
    def _initalize_solver(self, max_iter=10000) -> None:
        """ initializes the solver, max_iter and expand are set"""
        p_opts = {"expand": False}
        s_opts = {"max_iter": max_iter, "print_level": 5, 'fast_step_computation': "no"}
        self.opti.solver("blocksqp")

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
        placeholder1 = cs.DM.zeros((1, self.horizon))
        placeholder3 = cs.DM.zeros((3, self.horizon))
        return cs.vertcat(rdot, Vdot, qdot, placeholder3, placeholder1)

    def dynamics_mulitple(self, x: cs.MX, u: cs.MX, v:cs.MX) -> cs.MX:
        """ rk4 step for the dynamics, x is the state, u is the rotor rpm input"""
        k1 = self.derivative_multiple(x, u)
        k2 = self.derivative_multiple(x + (k1 * 0.5 * self.dt), u)
        k3 = self.derivative_multiple(x + (k2 * 0.5 * self.dt), u)
        k4 = self.derivative_multiple(x + k3 * self.dt, u)

        x_next = x + (k1 + k2 * 2 + k3 * 2 + k4) * self.dt / 6
        x_next[pos_theta, :] = x[pos_theta, :] + v * self.dt
        x_next[pos_omegab:pos_theta, :] = x_next[pos_omegab:pos_theta, :] + u[1:4, :] * self.dt
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

        ulist = np.empty((4, step_no))
        vlist = np.empty((1, step_no))
        xlist = [x0]

        self.opti.set_value(self.X0, x0)
        sol = self.opti.solve()

        # update state
        x0 = self.ode_dynamics(x0, self.torque_scaling(sol.value(self.U)[:, 0, None]), sol.value(self.V)[0], self.dt)
        x0[pos_q:pos_omegab] = Quaternion(x0[pos_q:pos_omegab]).normalized().wxyz

        # log variables
        ulist[:, 0:1] = self.torque_scaling(sol.value(self.U))[:, 0, None]
        xlist.append(x0)
        vlist[:, 0:1] = sol.value(self.V)[0]

        # linearize path
        line_start_location, line_dir = linearize_path(x0[pos_theta])
        self.opti.set_value(self.line_start_location, line_start_location)
        self.opti.set_value(self.line_start_along_path, x0[pos_theta])
        self.opti.set_value(self.line_dir, line_dir)

        # err_lag, err_con, vel = self._cost_function_new()
        # self.opti.minimize(err_lag + err_con - vel)

        end = time.time()
        timelist = np.append(timelist, end - start)
        for i in range(1, step_no):
            print(self.opti.debug.value(self.line_start_location))
            print(self.opti.debug.value(self.line_dir))
            print(self.opti.debug.value(self.line_start_along_path))
            print(x0)
            # calling the solver
            sol = self.optimization_step(x0, sol.value(self.X), sol.value(self.U), sol.value(self.V), False)
            # self.graph_infeasiblities(sol)

            # update position
            x0 = self.ode_dynamics(x0, self.torque_scaling(sol.value(self.U)[:, 0, None]), sol.value(self.V)[0],
                                   self.dt)
            x0[pos_q:pos_omegab] = Quaternion(x0[pos_q:pos_omegab]).normalized().wxyz

            ulist[:, i:i + 1] = self.torque_scaling(sol.value(self.U))[:, 0, None]
            xlist.append(x0)
            vlist[:, i:i + 1] = sol.value(self.V)[0]

            # linearize path
            line_start_location, line_dir = linearize_path(x0[pos_theta])
            self.opti.set_value(self.line_start_location, line_start_location)
            self.opti.set_value(self.line_start_along_path, x0[pos_theta])
            self.opti.set_value(self.line_dir, line_dir)

            timelist = np.append(timelist, time.time() - end)
            end = time.time()



        if return_x:
            return xlist, ulist, vlist, timelist
        else:
            return ulist, vlist, timelist

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

        return cs.vertcat(rdot, Vdot, qdot, 0, 0, 0, 0)

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
        k1 = MPCC_for_bodyrate.derivative_single(x, u)
        k2 = MPCC_for_bodyrate.derivative_single(x + (k1 * 0.5 * dt), u)
        k3 = MPCC_for_bodyrate.derivative_single(x + (k2 * 0.5 * dt), u)
        k4 = MPCC_for_bodyrate.derivative_single(x + k3 * dt, u)

        x_next = x + (k1 + k2 * 2 + k3 * 2 + k4) * dt / 6
        x_next[pos_theta] = x[pos_theta] + v * dt
        x_next[pos_omegab:pos_theta] = x[pos_omegab:pos_theta] + u[1:4] * dt

        return x_next

    @staticmethod
    def torque_scaling(u, torque_scaling_xx=0.02, torque_scaling_yy=0.02, torque_scaling_zz=0.02):
        return np.repeat(np.array([[1], [1 / torque_scaling_xx], [1 / torque_scaling_yy], [1 / torque_scaling_zz]]), u.shape[1], axis=1) * u
        # return np.diag([1, 1 / torque_scaling_xx, 1 / torque_scaling_yy, 1 / torque_scaling_zz]) @ u

    @staticmethod
    def V_scaling(v, v_scaling=1):
        return v * v_scaling

    @classmethod
    def ode_dynamics(cls, x0, u0, v0, dt):
        ode = {'x': state, 'u': input, 'ode': cls.derivative_single(state, input)}
        F = cs.integrator('F', 'cvodes', ode, {'t0': 0, 'tf': dt})
        res = F(x0=x0, u=u0)['xf']
        res[pos_theta] = res[pos_theta] + v0 * dt
        res[pos_omegab:pos_theta] = res[pos_omegab:pos_theta] + u0[1:4] * dt
        return res


