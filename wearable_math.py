import numpy as np


def generate_coordinate(points):
    origin = points[0]
    x_direction = points[1] - points[0]
    x_direction = x_direction / np.linalg.norm(x_direction)
    tmp = points[2] - points[0]
    y_direction = tmp - np.dot(tmp, x_direction)*x_direction
    y_direction = y_direction / np.linalg.norm(y_direction)
    z_direction = np.cross(x_direction, y_direction)
    z_direction = z_direction / np.linalg.norm(z_direction)
    return origin, x_direction, y_direction, z_direction


def get_relative_position(origin, x_axis, y_axis, z_axis, point):
    tmp = point - origin
    x = np.dot(tmp, x_axis)
    y = np.dot(tmp, y_axis)
    z = np.dot(tmp, z_axis)
    return np.array([x, y, z])


def get_world_position(origin, x_axis, y_axis, z_axis, relative_point):
    return origin + relative_point[0] * x_axis + relative_point[1] * y_axis + relative_point[2] * z_axis
