import numpy as np
import casadi as cs


class Quaternion:
    """Quaternions, order is wxyz not wxyz"""
    def __init__(self, wxyz):
        """Inputs:
            -wxyz: 4xN slicable matrix of quaternions"""
        self.wxyz = wxyz
        self.is_vector = False
        if wxyz.shape[0] == 3:
            self.wxyz = cs.vertcat(np.zeros((1, wxyz.shape[1])), wxyz)
            self.is_vector = True
        self.norm = cs.sqrt(cs.sum1(cs.power(wxyz, 2)))

    def __getitem__(self, item):
        return self.wxyz[item]

    def __mul__(self, other):
        """quaternion product
            Inputs:
                -other: 4xN slicable matrix of quaternions"""
        return cs.vertcat(self.wxyz[0, :] * other[0, :] - self.wxyz[1, :] * other[1, :] - self.wxyz[2, :] * other[2, :] - self.wxyz[3, :] * other[3, :],
                          self.wxyz[0, :] * other[1, :] + self.wxyz[1, :] * other[0, :] + self.wxyz[2, :] * other[3, :] - self.wxyz[3, :] * other[2, :],
                          self.wxyz[0, :] * other[2, :] - self.wxyz[1, :] * other[3, :] + self.wxyz[2, :] * other[0, :] + self.wxyz[3, :] * other[1, :],
                          self.wxyz[0, :] * other[3, :] + self.wxyz[1, :] * other[2, :] - self.wxyz[2, :] * other[1, :] + self.wxyz[3, :] * other[0, :])

    def conjugate(self):
        """conjugate of the quaternion"""
        conj = Quaternion(wxyz=self.wxyz)
        conj.wxyz[1:4] = -conj.wxyz[1:4]
        return conj

    def normalized(self):
        """normalize the quaternion"""
        return Quaternion(wxyz=self.wxyz/self.norm)

    def from_force(self, force, loc=3, depth=1):
        """quaternion from a force aligned with one of the axes"""
        above = cs.MX.zeros(loc-1, depth)
        below = cs.MX.zeros(3-loc, depth)
        return Quaternion(cs.vertcat(above, force, below))


class RotationQuaternion(Quaternion):
    """Quaternion but unit length"""
    def __init__(self, wxyz):
        """Inputs:
                    -wxyz: 4xN slicable matrix of quaternions"""
        self.wxyz = wxyz
        self.is_vector = False
        if wxyz.shape[1] == 3:
            self.wxyz = cs.vertcat(type(wxyz)(1, wxyz.shape[1]), wxyz)
            self.is_vector = True
        self.norm = cs.sqrt(cs.sum1(cs.power(wxyz, 2)))
        self.normalize()

    def normalize(self):
        """normalize the quaternion"""
        self.wxyz = cs.mtimes(self.wxyz, cs.power(cs.diag(self.norm), -1))


    def act(self, vector):
        """vector rotation by quaternion
            Inputs:
                -other: 3xN slicable matrix of vectors"""
        return (Quaternion(self * Quaternion(vector)) * self.conjugate())[1:4, :]

def create_vector_from_force(force, loc=3, depth=1):
    """quaternion from a force aligned with one of the axes"""
    above = cs.DM.zeros(loc-1, depth)
    below = cs.DM.zeros(3-loc, depth)
    return cs.vertcat(above, force, below)

def proj_transform(xs, ys, zs, M):
    """this is bc the projection transform used in the mplot3d was bugging out"""
    vec = np.squeeze(np.array([xs, ys, zs, np.ones_like(xs)]))
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here..
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    return txs, tys, tzs