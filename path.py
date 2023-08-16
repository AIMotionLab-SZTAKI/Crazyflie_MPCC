import casadi as cs
import numpy as np

distance = cs.SX.sym('distance', 1, 1)

# rn the path is only defined for a radius 1 circular helix
gyokketto = np.sqrt(2)
no_peaks = 4
steepness = 1
length_height_factor = np.sqrt(1 + steepness**2)  # d path per dz

steepness_func = cs.Function('steepness_func', [distance], [(cs.heaviside(cs.fmod(distance/2/np.pi * no_peaks / length_height_factor, 2) - 1)-0.5) * no_peaks / 4])

path_direction_func = cs.Function('direction_func', [distance], [cs.vertcat(-cs.sin(distance / length_height_factor),
                                                                            cs.cos(distance / length_height_factor),
                                                                            steepness_func(distance)) / length_height_factor])

path_position_func = cs.Function('position_func',
                                 [distance],
                                 [cs.vertcat(cs.cos(distance / length_height_factor),
                                  cs.sin(distance / length_height_factor),
                                  cs.power(cs.power(cs.fmod(distance / length_height_factor * no_peaks / (2 * np.pi), 2) - 1, 2), 0.5))])
