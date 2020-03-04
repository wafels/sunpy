#
# Calculates the co-ordinates along great arcs between two specified points
# which are assumed to be on disk.
#
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import Heliocentric


__all__ = ['GreatArc']


class GreatArc:
    """
    Calculate the properties of points on a great arc defined by two
    user-specified coordinates on a sphere.

    Parameters
    ----------
    initial : `~astropy.coordinates.SkyCoord`
        The first coordinate.

    target : `~astropy.coordinates.SkyCoord`
        The second coordinate.

    center : `~astropy.coordinates.SkyCoord`
        Center of the sphere.

    points : `None`, `int`, `numpy.ndarray`
        Number of points along the great arc.  If `None`, coordinates are
        calculated at 100 equally spaced points along the arc.  If `int`,
        coordinates are calculated at "points" equally spaced points along
        the arc.  If a numpy.ndarray is passed, it must be one dimensional
        and have values >=0 and <=1.  The values in this array correspond to
        parameterized locations along the arc, with zero corresponding to
        the initial coordinate and 1 corresponding to the last point of the
        arc. Setting this keyword on initializing a GreatArc object sets the
        locations of the default points along the great arc.

    great_circle : `bool`
        If True, calculate a great circle that passes through the initial and
        target coordinates.  If False, calculate points that lie along an arc
        between the initial and target coordinate.

    use_inner_angle_direction : `bool`
        Defines the direction of the great arc on the sphere. If True, then
        the great arc is directed along the inner angle from the initial to the
        target coordinate. If False, then the great arc is directed along the
        outer angle from the initial to the target coordinate.

    References
    ----------
    [1] https://www.mathworks.com/matlabcentral/newsreader/view_thread/277881
    [2] https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from sunpy.coordinates.utils import GreatArc
    >>> import sunpy.map
    >>> from sunpy.data.sample import AIA_171_IMAGE  # doctest: +REMOTE_DATA
    >>> m = sunpy.map.Map(AIA_171_IMAGE)  # doctest: +REMOTE_DATA
    >>> a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=m.coordinate_frame)  # doctest: +REMOTE_DATA
    >>> b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=m.coordinate_frame)  # doctest: +REMOTE_DATA
    >>> great_arc = GreatArc(a, b)  # doctest: +REMOTE_DATA
    >>> ax = plt.subplot(projection=m)  # doctest: +SKIP
    >>> m.plot(axes=ax)  # doctest: +SKIP
    >>> ax.plot_coord(great_arc.coordinates, color='c')  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    """
    def __init__(self, initial, target, center=None, points=100, great_circle=False, use_inner_angle_direction=True):
        # Initial and target coordinates
        self.initial = initial
        self.target = target

        # Observer
        self._output_observer = self.initial.observer

        # Co-ordinate frame of the initial coordinate is the frame used in the output
        self._output_frame = self.initial.frame

        # Observation time of the initial coordinate is the frame used in the output
        self._output_obstime = self.initial.obstime

        # Initial point of the great arc
        self._initial = self.initial.transform_to(Heliocentric)

        # Target point of the great arc
        self._target = self.target.transform_to(self._output_frame).transform_to(Heliocentric)

        # Units of the initial point
        self._distance_unit = self._initial.cartesian.xyz.unit

        # Set the center of the sphere
        if center is None:
            self._center = SkyCoord(0 * self._distance_unit,
                                    0 * self._distance_unit,
                                    0 * self._distance_unit,
                                    obstime=self._output_obstime,
                                    observer=self._output_observer,
                                    frame=Heliocentric)
        else:
            self._center = center.transform_to(self._output_frame).transform_to(Heliocentric)

        # Interpret the points keyword
        self.points = points
        if isinstance(self.points, int):
            self._points = np.linspace(0, 1, self.points)
        elif isinstance(self.points, np.ndarray):
            if self.points.ndim > 1:
                raise ValueError('One dimensional numpy ndarrays only.')
            if np.any(self.points < 0) or np.any(self.points > 1):
                raise ValueError('All value in points array must be strictly >=0 and <=1.')
            self._points = self.points
        else:
            raise ValueError('Incorrectly specified "points" keyword value.')

        # Did the user ask for a great circle?
        self.great_circle = great_circle

        # Which direction between the initial and target points?
        self.use_inner_angle_direction = use_inner_angle_direction

        # Convert the initial, target and center points to their Cartesian values
        self._initial_cartesian = self._initial.cartesian.xyz.to(self._distance_unit).value
        self._target_cartesian = self._target.cartesian.xyz.to(self._distance_unit).value
        self._center_cartesian = self._center.cartesian.xyz.to(self._distance_unit).value

        # Great arc properties calculation
        # Vector from center to first point
        self._v1 = self._initial_cartesian - self._center_cartesian

        # Distance of the first point from the center
        self._r = np.linalg.norm(self._v1)

        # Vector from center to second point
        self._v2 = self._target_cartesian - self._center_cartesian

        # The v3 vector lies in plane of v1 & v2 and is orthogonal to v1
        self._v3 = np.cross(np.cross(self._v1, self._v2), self._v1)
        self._v3 = self._r * self._v3 / np.linalg.norm(self._v3)

        # Radius of the sphere
        self._radius = self._r * self._distance_unit

        # Calculate the angle subtended by the requested arc
        if self.great_circle:
            full_circle = 2 * np.pi * u.rad
            if self.use_inner_angle_direction:
                self._angle = full_circle
            else:
                self._angle = -full_circle
        else:
            # Inner angle between v1 and v2 in radians
            inner_angle = np.arctan2(np.linalg.norm(np.cross(self._v1, self._v2)), np.dot(self._v1, self._v2)) * u.rad
            if self.use_inner_angle_direction:
                self._angle = inner_angle
            else:
                self._angle = inner_angle - 2 * np.pi * u.rad

        # Distance on the sphere between the initial point and the target point.
        self._distance = self._radius * self._angle.value

        # Interpret the points keyword
        self.points = points
        if isinstance(self.points, int):
            self._points = np.linspace(0, 1, self.points)
        elif isinstance(self.points, np.ndarray):
            if self.points.ndim > 1:
                raise ValueError('One dimensional numpy ndarrays only.')
            if np.any(self.points < 0) or np.any(self.points > 1):
                raise ValueError('All value in points array must be strictly >=0 and <=1.')
            self._points = self.points
        else:
            raise ValueError('Incorrectly specified "points" keyword value.')

        # Angles needed for the coordinate calculation
        self._angles = self._points.reshape(len(self._points), 1)*self._angle.value

    @property
    def angles(self):
        """
        The angles subtended by the arc coordinates relative to the initial
        coordinate.
        """
        return self._points * self._angle

    @property
    def distances(self):
        """
        The distance along the sphere for all the coordinates relative to the
        initial coordinate.
        """
        return self._radius * self.angles.value

    @property
    def coordinates(self):
        """
        Calculates the co-ordinates of the arc, returned in the frame of
        the start coordinate.

        Returns
        -------
        coordinates : `~astropy.coordinates.SkyCoord`
            Co-ordinates along the great arc in the co-ordinate frame of the
            initial coordinate.
        """

        # Calculate the Cartesian locations from the first to second points
        great_arc_points_cartesian = (self._v1[np.newaxis, :] * np.cos(self._angles) +
                                      self._v3[np.newaxis, :] * np.sin(self._angles) +
                                      self._center_cartesian) * self._distance_unit

        # Return the coordinates of the great arc between the start and end
        # points
        return SkyCoord(great_arc_points_cartesian[:, 0],
                        great_arc_points_cartesian[:, 1],
                        great_arc_points_cartesian[:, 2],
                        obstime=self._output_obstime,
                        observer=self._output_observer,
                        frame=Heliocentric).transform_to(self._output_frame)


class ArcVisibility:
    def __init__(self, coordinates, great_circle=False):
        self.coordinates = coordinates
        self.great_circle = great_circle
        self._visibility = self.coordinates.transform_to(Heliocentric).z.value > 0
        self._front = self._visibility.astype(np.int)
        self._change = self._front[1:] - self._front[0:-1]

    @property
    def visibility(self):
        """
        The visibility of each of the arc coordinates as seen from the arc's
        observer.  If a value is True, then the coordinate is in front of the
        plane of the sky.  If False, then the coordinate is behind the plane of
        the sky.
        """
        return self._visibility

    @property
    def all_on_front(self):
        """
        Returns True if every arc coordinate is visible, False otherwise.  When
        True, every arc coordinate is visible from the arc's observer.
        """
        return np.all(self._visibility)

    @property
    def all_on_back(self):
        """
        Returns True if every arc coordinate is not visible, False otherwise. When
        True, every coordinate is on the back of the Sun as seen from the arc's
        observer.
        """
        return np.all(~self._visibility)

    @property
    def from_front_to_back(self):
        """
        Returns the index at which the visibility of the arc switches from being
        visible (on the disk of the Sun) to not visible (on the back of the Sun),
        as seen by the arc's observer.  The index indicates the first coordinate
        along the direction of the arc that is not visible from the arc's observer.
        If all the arc coordinates are on the front or back of the visible disk of
        the Sun as seen from the arc's viewpoint, None is returned.
        """
        if self.all_on_front or self.all_on_back:
            return None
        else:
            test = self._change == -1
            if np.any(test):
                return np.where(test)[0][0]
            else:
                return None

    @property
    def from_back_to_front(self):
        """
        Returns the index at which the visibility of the arc switches from being
        not visible (on the back of the Sun) to visible (on the front of the Sun),
        as seen by the arc's observer.  The index indicates the first coordinate
        along the direction of the arc that is visible from the arc's observer.
        If all the arc coordinates are on the front or back of the visible disk of
        the Sun as seen from the arc's viewpoint, None is returned.
        """
        if self.all_on_front or self.all_on_back:
            return None
        else:
            test = self._change == 1
            if np.any(test):
                return np.where(test)[0][0]
            else:
                return None

    @property
    def great_circle_front_indices(self):
        """
        Returns a set of indices for the input coordinates that describe a continuous arc on the Sun that is entirely on
        the front of the disk of the Sun for great circles.  If the object is created with great_circle=False then no
        indices are returned.
        """
        if self.great_circle:
            logic = np.roll(self.visibility, -self.from_back_to_front - 1)
            indices = np.roll(np.arange(0, len(self.visibility)), -self.from_back_to_front - 1)
            return indices[logic]
        else:
            return None

    @property
    def great_circle_back_indices(self):
        """
        Returns a set of indices for the input coordinates that describe a continuous arc on the Sun that is entirelt on
        the front of the disk of the Sun for great circles.  If the object is created with great_circle=False then no
        indices are returned.
        """
        if self.great_circle:
            logic = np.roll(self.visibility, -self.from_front_to_back - 1)
            indices = np.roll(np.arange(0, len(self.visibility)), -self.from_back_to_front - 1)
            return indices[logic]
        else:
            return None

