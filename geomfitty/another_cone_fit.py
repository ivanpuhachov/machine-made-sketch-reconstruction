"""
https://github.com/JohnRomanelis/SHREC2022_PrimitiveRecognition/blob/master/cone_fit.py
"""

from mimetypes import init
from turtle import distance
import numpy as np
import numpy.linalg as la
from scipy import optimize  # type: ignore

from . import geom3d
from .fit3d import _check_input
from ._util import distance_line_point, distance_plane_point, distance_point_point
from ._descriptor import Direction, Position, PositiveNumber


class Cone(geom3d.GeometricShape):
    vertex = Position(3)
    axis = Direction(3)

    # theta = PositiveNumber()

    def __init__(self, theta, axis, vertex):
        self.vertex = vertex
        self.axis = axis
        self.theta = theta

    def __repr__(self):
        return f"Cone (vertex={self.vertex}, axis={self.axis}, theta={self.theta}"

    def distance_to_point(self, point):
        a = distance_line_point(self.vertex, self.axis, point)
        k = a * np.tan(self.theta)
        b = k + np.abs(np.dot((point - self.vertex), self.axis))
        l = b * np.sin(self.theta)
        d = a / np.cos(self.theta) - l  # np.abs

        return np.abs(d)


def cone_fit_residuals(cone_params, points, weights):

    cone = Cone(cone_params[0], cone_params[1:4], cone_params[4:7])

    distances = cone.distance_to_point(points)

    if weights is None:
        return distances

    return distances * np.sqrt(weights)


def cone_fit_residuals_split(
        theta,
        axis,
        vertex,
        points,
        weights=None,
):

    cone = Cone(
        theta=theta,
        axis=axis,
        vertex=vertex,
    )

    distances = cone.distance_to_point(points)

    if weights is None:
        return distances

    return distances * np.sqrt(weights)


def cone_fit(points, weights=None, initial_guess: Cone = None):
    """Fits a cone through a set of points"""
    _check_input(points, weights)
    if initial_guess is None:
        raise NotImplementedError

    # print(initial_guess.theta, initial_guess.axis.shape)
    x0 = np.concatenate([np.array([initial_guess.theta]), initial_guess.axis, initial_guess.vertex])
    results = optimize.least_squares(
        cone_fit_residuals, x0=x0, args=(points, weights), ftol=1e-10
    )

    if not results.success:
        return RuntimeError(results.message)

    cone = Cone(results.x[0], results.x[1:4], results.x[4:7])

    return cone
