import math

import numpy as np
import quaternionic


class state:
    def __init__(self, r=np.array([0, 0, 0]), V=np.array([0, 0, 0]), q=quaternionic.array([0, 0.5, 0.5, 0.70710678118]),
                 omegaB=np.array([0, 0, 0])):
        self.r = r
        self.V = V
        self.q = q
        self.omegaB = omegaB

    def __add__(self, other):
        return state(self.r + other.r, self.V + other.V, self.q + other.q, self.omegaB + other.omegaB)

    def __sub__(self, other):
        return state(self.r - other.r, self.V - other.V, self.q - other.q, self.omegaB - other.omegaB)

    def __mul__(self, other):
        return state(self.r * other, self.V * other, self.q * other, self.omegaB * other)

    def __truediv__(self, other):
        return state(self.r / other, self.V / other, self.q / other, self.omegaB / other)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __str__(self):
        return f"r: {self.r}\nV: {self.V}\nq: {self.q}\nomegaB: {self.omegaB}"

    def get_eulerangles(self):
        return self.q.to_euler_angles

    def normalize_quaternion(self):
        self.q = self.q.normalized


class drone:
    def __init__(self, V0=np.array([0, 0, 0])):
        self.mass = 0.028 / 1000  # kg
        self.gravitational_accel = 9.80665  # m/s^2
        self.J = np.array([[1.4e-5, 0, 0], [0, 1.4e-5, 0], [0, 0, 2.17e-5]])  # kg*m^2
        self.J_inv = np.linalg.inv(self.J)
        self.thrustCoefficient = 2.88e-8  # N*s^2
        self.dragCoefficient = 7.24e-10  # N*m*s^2
        self.propdist = 0.092
        self.dt = 0.05

        self.statevector = state(V=V0)

        #most receent forces and torques
        self.Thrust = 0  # N
        self.Torque = np.array([0, 0, 0])  # N*m

        #conversion matrix from rotor rpms to forces and torques
        a = np.array([[1], [self.propdist/np.sqrt(2)], [self.propdist/np.sqrt(2)], [self.dragCoefficient/self.thrustCoefficient]])
        b = np.array([[1, 1, 1,1 ], [-1, -1, 1, 1], [1, -1, -1, 1], [1, -1, 1, -1]])
        self.input_conversion_matrix = np.matmul(a, b)


    def derival(self, statevector=None):
        if statevector is None:
            statevector = self.statevector

        rdot = statevector.V
        Vdot = statevector.q.rotate(np.array([0, 0, -self.Thrust])) / self.mass - np.array(
            [0, 0, self.gravitational_accel])
        qdot = 0.5 * statevector.q * quaternionic.array.from_vector_part(statevector.omegaB)
        omegaBdot = np.matmul(self.J_inv,
                              (self.Torque - np.cross(statevector.omegaB, np.matmul(self.J, statevector.omegaB))))

        return state(rdot, Vdot, qdot, omegaBdot)  # the derivative of the state is a state itself

    def update(self, statevector=None, input=np.array([0, 0, 0, 0])):
        if statevector is None:
            statevector = self.statevector


        self.set_forces(input)
        k1 = self.derival()
        k2 = self.derival(statevector + (k1 * self.dt * 0.5))
        k3 = self.derival(statevector + (k2 * self.dt * 0.5))
        k4 = self.derival(statevector + k3 * self.dt)

        self.statevector = self.statevector + (k1 + k2 * 2 + 2 * k3 + k4)/6
        return self.statevector


    def set_forces(self, rpms):
        a = np.matmul(self.input_conversion_matrix, self.thrustCoefficient * rpms ** 2)
        self.Thrust = a[0]
        self.Torque = a[1:4]



if __name__ == "__main__":
    drone = drone()
    for i in range(4):
        print(drone.statevector)
        print("euler anlges: ", drone.statevector.get_eulerangles(), "\n")
        rpms = np.array([100, 100, 100, 100])
        drone.update()