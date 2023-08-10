import casadi as cs
import numpy as np

distance = cs.MX.sym('distance', 1, 1)

# rn the path is only defined for a radius 1 circular helix
gyokketto = np.sqrt(2)

steepness = 1
length_height_factor = np.sqrt(1 + steepness**2)  # d path per dz

path_direction_func = cs.Function('direction_func', [distance], [cs.vertcat(-cs.sin(distance / length_height_factor),
                                                                            cs.cos(distance / length_height_factor),
                                                                            steepness) / length_height_factor])

path_position_func = cs.Function('position_func', [distance], [cs.vertcat(cs.cos(distance / length_height_factor),
                                                                         cs.sin(distance / length_height_factor),
                                                                         steepness * distance / length_height_factor)])
