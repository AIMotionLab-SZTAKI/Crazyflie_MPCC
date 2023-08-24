import casadi as cs
import numpy as np
import scipy as cp
import matplotlib.pyplot as plt


distance = cs.SX.sym('distance', 1, 1)

# rn the path is only defined for a radius 1 circular helix
gyokketto = np.sqrt(2)
no_sharp_bends = 4
steepness = no_sharp_bends / 2 / np.pi # for each sharp bend the vertical distance traveled increases by 1 through one revolution
length_height_factor = np.sqrt(1 + steepness ** 2)  # d path per dz

# steepness_func = cs.Function('steepness_func', [distance], [(cs.heaviside(cs.fmod(distance / 2 / np.pi * no_sharp_bends / length_height_factor, 2) - 1) - 0.5) * no_sharp_bends / 4]) # idk ez adta ki a jo megoldást3
#
# path_direction_func = cs.Function('direction_func', [distance], [cs.vertcat(-cs.sin(distance / length_height_factor),
#                                                                             cs.cos(distance / length_height_factor),
#                                                                             steepness_func(distance)) / length_height_factor])
#
# path_position_func = cs.Function('position_func', [distance],
#                                  [cs.vertcat(cs.cos(distance / length_height_factor),
#                                   cs.sin(distance / length_height_factor),
#                                   cs.power(cs.power(cs.fmod(distance / length_height_factor * no_sharp_bends / (2 * np.pi), 2) - 1, 2), 0.5))])

steepness = 1
length_height_factor = np.sqrt(1 + steepness ** 2)  # d path per dz
path_direction_func = cs.Function('direction_func', [distance], [cs.vertcat(-cs.sin(distance / length_height_factor),
                                                                            cs.cos(distance / length_height_factor),
                                                                            steepness) / length_height_factor])

path_position_func = cs.Function('position_func', [distance], [cs.vertcat(cs.cos(distance / length_height_factor),
                                                                         cs.sin(distance / length_height_factor),
                                                                        steepness * distance / length_height_factor)])

def linearize_path(distance):
    """Returns the linearized path at the given distance"""
    kd1 = np.array(path_direction_func(distance + 0.1).elements())
    kd2 = np.array(path_direction_func(distance + 0.2).elements())
    kd3 = np.array(path_direction_func(distance + 0.3).elements())
    norm = np.linalg.norm((kd1+kd2*2+kd3))
    return path_position_func(distance), (kd1+kd2*2+kd3)/norm


class Spline_3D:
    def __init__(self, points: np.array, bc_type: str = 'periodic'):
        """

        :param points: Points to which the time-parametrized spline is interpolated
        :param bc_type: Type of boundary condition
        """

        self.shape = np.shape(points)
        self.original_points = points
        self.equally_space_points = None

        self.spl_t = cp.interpolate.CubicSpline(points[:, 0], points[:, 1:], bc_type=bc_type)
        self.__find_arclength_par()

    def __find_arclength_par(self, m: int = 60, e: float = 10**(-3), dt: float = 10**(-4)):
        """Class method that finds approximation of the arc length parametrisation of the time-parameterized spline

        :param m: Number of sections
        :param e: Error margin of bisection method
        :param dt: Precision of numerical integration
        """

        # Calculation of the total arc length
        t = np.arange(min(self.original_points[:, 0]), max(self.original_points[:, 0]) + dt, dt)
        dxydt = self.spl_t.derivative(1)(t)  # calculating the derivative of the time-parametrized spline

        ds = np.sqrt(dxydt[:, 0]**2 + dxydt[:, 1]**2 + dxydt[:, 2]**2)  # Length of arc element
        self.L = cp.integrate.simpson(y=ds, dx=dt)  # Simpsons 3/4 rule
        print(self.L)
        # Splitting the spline into m sections with length l using bisection
        self.l = self.L/m

        # Initializing bisection method
        tstart = min(self.original_points[:, 0])
        tend = max(self.original_points[:, 0])
        tmid = (tstart+tend)/2
        t_arr = np.array([0])
        self.s_arr = np.array([0])
        s_mid = 10000000

        # Solving problem with bisection
        for i in range(1, m):
            if i != 1:
                tstart = tmid
                tend = max(self.original_points[:, 0])
                tmid = (tstart + tend) / 2
                s_mid = 10000000

            while abs(s_mid-self.l) >= e:

                tmid_arr = np.arange(t_arr[-1], tmid + dt, dt)
                grad_mid = self.spl_t.derivative(1)(tmid_arr)
                ds_mid = np.sqrt(grad_mid[:, 0] ** 2 + grad_mid[:, 1] ** 2 + grad_mid[:, 2] ** 2)
                s_mid = cp.integrate.simpson(y=ds_mid, dx=dt)

                if self.l < s_mid:
                    tend = tmid
                    tmid = (tend+tstart)/2
                else:
                    tstart = tmid
                    tmid = (tend + tstart) / 2
            self.s_arr = np.append(self.s_arr, s_mid+i*self.l)
            t_arr = np.append(t_arr, tmid)
            #print(self.s_arr[-1]-(i+1)*self.l)

        self.s_arr = np.reshape(self.s_arr, (-1, 1))

        self.equally_space_points = np.concatenate((self.s_arr, self.spl_t(t_arr)), 1)  # array that contains the new points
        if (self.original_points[0, 1:] == self.original_points[-1, 1:]).all():
            self.equally_space_points = np.concatenate((self.equally_space_points, [[self.L+self.l, self.original_points[-1, 1], self.original_points[-1, 2], self.original_points[-1, 3]]]))


        self.spl_sx = cs.interpolant('n', 'bspline', [self.equally_space_points[:, 0]], self.equally_space_points[:, 1])  # fitting casadi spline to the x coordinate
        self.spl_sy = cs.interpolant('n', 'bspline', [self.equally_space_points[:, 0]], self.equally_space_points[:, 2])  # fitting casadi spline to the y coordinate
        self.spl_sz = cs.interpolant('n', 'bspline', [self.equally_space_points[:, 0]], self.equally_space_points[:, 3])  # fitting casadi spline to the z coordinate

    def get_path_parameters(self, theta, theta_0=None):
        """
        Path parameters using vectors
        :param theta:
        :param theta_0:
        :return:
        """
        point = cs.hcat((self.spl_sx(theta), self.spl_sy(theta), self.spl_sz(theta))).T

        jac_x = self.spl_sx.jacobian()
        jac_y = self.spl_sy.jacobian()
        jac_z = self.spl_sz.jacobian()

        v = cs.hcat((jac_x(theta, theta), jac_y(theta, theta), jac_z(theta, theta)))  # unit direction vector
        #l = v**2
        #v = cs.hcat((v[:, 0]/cs.sqrt(l[:,0]+l[:,1]), v[:, 1]/cs.sqrt(l[:,0]+l[:,1])))#cs.hcat((cs.sqrt(l[:, 0]+l[: 1]), cs.sqrt(l[:, 0]+l[: 1])))
        return point, v.T

    def get_path_parameters_lin(self, theta, theta_0):
        """
        Path parameters using first order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        x_0 = self.spl_sx(theta_0)
        y_0 = self.spl_sy(theta_0)
        z_0 = self.spl_sz(theta_0)

        jac_x = self.spl_sx.jacobian()
        jac_x = jac_x(theta_0, theta_0)
        jac_y = self.spl_sy.jacobian()
        jac_y = jac_y(theta_0, theta_0)
        jac_z = self.spl_sz.jacobian()
        jac_z = jac_z(theta_0, theta_0)

        x_lin = x_0 + jac_x * (theta - theta_0)
        y_lin = y_0 + jac_y * (theta - theta_0)
        z_lin = z_0 + jac_z * (theta - theta_0)

        point = cs.hcat((x_lin, y_lin, z_lin)).T
        v = cs.hcat((jac_x, jac_y, jac_z))/cs.sqrt(jac_x**2+jac_y**2+jac_z**2)

        return point, v.T

    def get_path_parameters_quad(self, theta, theta_0):
        """
        Path parameters using second order Taylor
        :param theta:
        :param theta_0:
        :return:
        """
        x_0 = self.spl_sx(theta_0)
        y_0 = self.spl_sy(theta_0)
        z_0 = self.spl_sz(theta_0)

        jac_x = self.spl_sx.jacobian()
        jac2_x = jac_x.jacobian()
        jac_x = jac_x(theta_0, theta_0)
        jac2_x = jac2_x(theta_0, theta_0, theta_0)[:,1]

        jac_y = self.spl_sy.jacobian()
        jac2_y = jac_y.jacobian()
        jac_y = jac_y(theta_0, theta_0)
        jac2_y = jac2_y(theta_0, theta_0, theta_0)[:,1]

        jac_z = self.spl_sz.jacobian()
        jac2_z = jac_z.jacobian()
        jac_z = jac_z(theta_0, theta_0)
        jac2_z = jac2_z(theta_0, theta_0, theta_0)[:, 1]

        x_lin = x_0 + jac_x * (theta - theta_0) + jac2_x/2 * (theta - theta_0)**2
        y_lin = y_0 + jac_y * (theta - theta_0) + jac2_y/2 * (theta - theta_0)**2
        z_lin = z_0 + jac_z * (theta - theta_0) + jac2_z / 2 * (theta - theta_0)**2

        point = cs.hcat((x_lin, y_lin, z_lin)).T

        jac_x_lin = jac_x + jac2_x * (theta - theta_0)
        jac_y_lin = jac_y + jac2_y * (theta - theta_0)
        jac_z_lin = jac_z + jac2_z * (theta - theta_0)

        v = cs.hcat((jac_x_lin, jac_y_lin, jac_z_lin))
        l = v ** 2
        v = cs.hcat((v[:, 0]/cs.sqrt(l[:,0]+l[:,1]+l[:,2]), v[:, 1]/cs.sqrt(l[:,0]+l[:,1]+l[:,2]), v[:, 2]/cs.sqrt(l[:,0]+l[:,1]+l[:,2])))
        return point, v.T

    def plot_original(self):
        t = np.linspace(self.original_points[0, 0], self.original_points[-1, 0], 200)
        coords = self.spl_t(t)

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])
        plt.show()

    def plot_equally(self):
        s = np.linspace(self.equally_space_points[0, 0], self.equally_space_points[-1, 0], 200)
        x = self.spl_sx(s)
        y = self.spl_sy(s)
        z = self.spl_sz(s)

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x, y, z)
        ax.scatter(self.equally_space_points[:, 1], self.equally_space_points[:, 2], self.equally_space_points[:,3])

if __name__ == "__main__":
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
    pos = path_spline.get_path_parameters(5.32)
    print(pos)
    plt.show()