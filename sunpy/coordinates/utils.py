#
# Calculates the co-ordinates along great arcs between two specified points
# which are assumed to be on disk.
#
from __future__ import absolute_import, division, print_function

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import frames

__all__ = ['inner_angle', 'GreatArc', 'GreatCircle']


#
# Calculates the inner angle along a great arc between two specified points on
# the solar sphere.
#
def inner_angle(start, end, center=None):
        """

        :param start:
        :param end:
        :param center:
        :return:
        """
        # Units of the start point
        distance_unit = start.transform_to(frames.Heliocentric).cartesian.xyz.unit

        # Set the center of the sphere
        if center is None:
            c = SkyCoord(0 * distance_unit, 0 * distance_unit, 0 * distance_unit, frame=frames.Heliocentric)

        # Convert the start, end and center points to their Cartesian values
        v1 = _skycoord_to_cartesian_vector_equivalent(start, c, distance_unit)
        v2 = _skycoord_to_cartesian_vector_equivalent(end, c, distance_unit)

        return _inner_angle(v1, v2) * u.rad


# Inner angle between vector v1 and v2 in radians
def _inner_angle(v1, v2):
    """

    :param v1:
    :param v2:
    :return:
    """
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


# Calculate the Cartesian equivalent of an input SkyCoord
def _skycoord_to_cartesian(skycoord, distance_unit):
    """

    :param skycoord:
    :param distance_unit:
    :return:
    """
    return skycoord.transform_to(frames.Heliocentric).cartesian.xyz.to(distance_unit).value


def _skycoord_to_cartesian_vector_equivalent(skycoord, center, distance_unit):
    """

    :param skycoord:
    :param center:
    :param distance_unit:
    :return:
    """
    return _skycoord_to_cartesian(skycoord, distance_unit) - _skycoord_to_cartesian(center, distance_unit)


class GreatArc(object):
    """
    Calculate the properties of a great arc at user-specified points between a
    start and end point on a sphere.

    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        Start point.

    end : `~astropy.coordinates.SkyCoord`
        End point.

    center : `~astropy.coordinates.SkyCoord`
        Center of the sphere.

    points : `int`, `~numpy.ndarray`
        Number of points along the great arc.  If points is not explicitly set,
        the arc is calculated at 100 equally spaced points from start to end.
        If int, the arc is calculated at "points" equally spaced points from
        start to end.  If a numpy.ndarray is passed, it must be one dimensional
        and have values >=0 and <=1.  The values in this array correspond to
        parameterized locations along the great arc from 0, denoting the start
        of the arc, to 1, denoting the end of the arc.

    Properties
    ----------
    The GreatArc object has a number of useful properties.

    inner_angle : `astropy.units.rad`
        The inner angle between the start and end point in radians.

    distance :
        Distance on the sphere between the start point and the end point.

    radius :
        Radius of the sphere.

    inner_angles : `astropy.units.rad`
        The inner angles for the parameterized points along the arc from the
        start co-ordinate to the end measured in radius.

    distances :
        Distance on the sphere of the parameterized points along the arc from
        the start co-ordinate to the end measured in radius.

    cartesian_coordinates : `~astropy.coordinates.SkyCoord`
        Co-ordinates along the great arc in the Cartesian co-ordinate frame for
        the parameterized points along the arc.

    coordinates : `~astropy.coordinates.SkyCoord`
        Co-ordinates along the great arc in the co-ordinate frame of the start
        point for the parameterized points along the arc.

    front_or_back : `~numpy.ndarray`
        Boolean array indicating which coordinates are on the front or the back
        of the sphere as seen by the observer.

    front_arc_indices: `~numpy.ndarray`


    back_arc_indices : `~numpy.ndarray`


    from_front_to_back_index : `int`


    from_back_to_front_index : `int`


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
    >>> from sunpy.data.sample import AIA_171_IMAGE
    >>> m = sunpy.map.Map(AIA_171_IMAGE)
    >>> a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=m.coordinate_frame)
    >>> b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=m.coordinate_frame)
    >>> great_arc = GreatArc(a, b)
    >>> ax = plt.subplot(projection=m)
    >>> m.plot(axes=ax)
    >>> ax.plot_coord(great_arc.coordinates, color='c')
    >>> plt.show()
    """

    def __init__(self, start, end, center=None, points=np.linspace(0, 1, 100)):
        # Start point of the great arc
        self.start = start

        # End point of the great arc
        self.end = end

        # Interpret the user parameterized points request
        if isinstance(points, int):
            self.points = np.linspace(0, 1, points)
        elif isinstance(points, np.ndarray):
            if points.ndim > 1:
                raise ValueError('One dimensional numpy ndarrays only.')
            if np.any(points < 0) or np.any(points > 1):
                raise ValueError('All value in points array must be strictly >=0 and <=1.')
            self.points = points
        else:
            raise ValueError('Incorrectly specified "points" keyword value.')

        # Number of points
        self._npoints = len(self.points)

        # Units of the start point
        self._distance_unit = self.start.transform_to(frames.Heliocentric).cartesian.xyz.unit

        # Co-ordinate frame
        self._start_frame = self.start.frame

        # Observer
        self._observer = self.start.observer

        # Set the center of the sphere
        if center is None:
            self.center = SkyCoord(0 * self._distance_unit,
                                   0 * self._distance_unit,
                                   0 * self._distance_unit, frame=frames.Heliocentric)

        # Convert the start, end and center points to their Cartesian values
        self._center_cartesian = _skycoord_to_cartesian(self.center, self._distance_unit)
        self._v1 = _skycoord_to_cartesian_vector_equivalent(self.start, self.center, self._distance_unit)
        self._v2 = _skycoord_to_cartesian_vector_equivalent(self.end, self.center, self._distance_unit)

        # Distance of the first point from the center
        self._r = np.linalg.norm(self._v1)

        # The v3 vector lies in plane of v1 & v2 and is orthogonal to v1
        self._v3 = np.cross(np.cross(self._v1, self._v2), self._v1)
        self._v3 = self._r * self._v3 / np.linalg.norm(self._v3)

        # Inner angle between v1 and v2 in radians
        self.inner_angle = inner_angle(self.start, self.end, center=self.center)

        # Radius of the sphere
        self.radius = self._r * self._distance_unit

        # Distance on the sphere between the start point and the end point.
        self.distance = self.radius * self.inner_angle.value

        # Calculates the inner angles for the parameterized points along the
        # arc and returns the value in radians, from the start co-ordinate to
        # the end.
        self.inner_angles = self.points.reshape(self._npoints, 1)*self.inner_angle

        # Distances of the points along the great arc from the start to end
        # co-ordinate.  The units are defined as those returned after
        # transforming the co-ordinate system of the start co-ordinate into
        # its Cartesian equivalent.
        self.distances = self.radius * self.inner_angles.value

        # Cartesian coordinates for the great arc.
        self.cartesian_coordinates = self._calculate_great_arc_cartesian_skycoord(self._v1[np.newaxis, :],
                                                                                  self._v3[np.newaxis, :],
                                                                                  self.inner_angles)

        # Co-ordinates along the great arc in the co-ordinate frame of the
        # start point.
        self.coordinates = self.cartesian_coordinates.transform_to(self._start_frame)

        # Boolean array indicating which coordinates are on the front of the
        # disk (True) or on the back (False).
        self.front_or_back = self.cartesian_coordinates.z.value >= 0

        # Boolean properties that describe if the arc is all on the front side
        # or all on the backside.
        if np.all(self.front_or_back):
            self.all_on_front = True
        else:
            self.all_on_front = False

        if np.all(~self.front_or_back):
            self.all_on_back = True
        else:
            self.all_on_back = False

        # Calculate the indices where the co-ordinates change from being on the
        # front of the disk to the back to the disk.
        if self.all_on_front:
            self.from_front_to_back_points_index = None
            self.from_back_to_front_points_index = None
            self.front_arc_indices = np.arange(0, self._npoints)
            self.back_arc_indices = None

        if self.all_on_back:
            self.from_front_to_back_points_index = None
            self.from_back_to_front_points_index = None
            self.front_arc_indices = None
            self.back_arc_indices = np.arange(0, self._npoints)

        if not self.all_on_back and not self.all_on_front:
            self._fob = self.front_or_back.astype(np.int)
            self._change = self._fob[1:] - self._fob[0:-1]
            self.from_front_to_back_points_index = np.where(self._change == -1)[0][0]
            self.from_back_to_front_points_index = np.where(self._change == 1)[0][0]

            # Indices of arc on the front side
            self.front_arc_indices = np.concatenate((np.arange(self.from_back_to_front_points_index, self._npoints),
                                                     np.arange(0, self.from_front_to_back_points_index)))

            # Indices of arc on the back side
            self.back_arc_indices = np.arange(self.from_front_to_back_points_index + 1,
                                              self.from_back_to_front_points_index)

        # Calculate the exact location where a great circle that runs from the
        # start coordinate to the end coordinate crosses the disk.
        angle1 = np.arctan2(-self._v1[2], self._v3[2]) * u.rad
        angle2 = np.mod(angle1 + np.pi, 2 * np.pi)
        if self.front_or_back[0]:
            # First point is on the front
            self.from_front_to_back_coordinate = self._calculate_great_arc_cartesian_skycoord(self._v1, self._v3, np.min([angle1, angle2]))
            self.from_back_to_front_coordinate = self._calculate_great_arc_cartesian_skycoord(self._v1, self._v3, np.max([angle1, angle2]))
        else:
            # First point is on the back
            self.from_front_to_back_coordinate = self._calculate_great_arc_cartesian_skycoord(self._v1, self._v3, np.max([angle1, angle2]))
            self.from_back_to_front_coordinate = self._calculate_great_arc_cartesian_skycoord(self._v1, self._v3, np.min([angle1, angle2]))

    # Co-ordinates along the great arc in the Cartesian co-ordinate frame.
    def _calculate_great_arc_cartesian_skycoord(self, v1, v3, angles):
        great_arc_points_cartesian = (v1 * np.cos(angles) + v3 * np.sin(angles) + self._center_cartesian) * self._distance_unit
        return SkyCoord(great_arc_points_cartesian[:, 0],
                        great_arc_points_cartesian[:, 1],
                        great_arc_points_cartesian[:, 2],
                        frame=frames.Heliocentric,
                        observer=self._observer)


class GreatCircle(GreatArc):
    """
    Calculate the properties of a great circle at user-specified points
    The great circle passes through two points on a sphere specified by
    the user.  The points returned are in the direction from the start point
    through the end point.

    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        Start point.

    end : `~astropy.coordinates.SkyCoord`
        End point.

    center : `~astropy.coordinates.SkyCoord`
        Center of the sphere.

    points : `int`, `~numpy.ndarray`
        Number of points along the great circle.   If points is not
        explicitly set, the circle is calculated at 100 equally spaced points
        from start to end. If int, the circle is calculated at "points" equally
        spaced points from start to end.  If a numpy.ndarray is passed, it must
        be one dimensional and have values >=0 and <=1.  The values in this
        array correspond to parameterized locations along the great circle from
        0, denoting the start of the circle, to 1, denoting the end of the
        circle.

    Properties
    ----------
    The GreatCircle object has the same properties as the GreatArc object.
    Please refer to the documentation for the GreatArc object.

    References
    ----------
    [1] https://www.mathworks.com/matlabcentral/newsreader/view_thread/277881
    [2] https://en.wikipedia.org/wiki/Great-circle_distance#Vector_version

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from sunpy.coordinates.utils import GreatCircle
    >>> import sunpy.map
    >>> from sunpy.data.sample import AIA_171_IMAGE
    >>> m = sunpy.map.Map(AIA_171_IMAGE)
    >>> a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=m.coordinate_frame)
    >>> b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=m.coordinate_frame)
    >>> great_circle = GreatCircle(a, b, points=1000)
    >>> coordinates = great_circle.coordinates
    >>> front_arc_coordinates = coordinates[great_circle.front_arc_indices]
    >>> back_arc_coordinates = coordinates[great_circle.back_arc_indices]
    >>> arc_from_start_to_back = coordinates[0:great_circle.from_front_to_back_index]
    >>> arc_from_back_to_start = coordinates[great_circle.from_back_to_front_index: len(coordinates)-1]
    >>> ax = plt.subplot(projection=m)
    >>> m.plot(axes=ax)
    >>> ax.plot_coord(front_arc_coordinates, color='k', linewidth=5)
    >>> ax.plot_coord(back_arc_coordinates, color='k', linestyle=":")
    >>> ax.plot_coord(arc_from_start_to_back, color='c')
    >>> ax.plot_coord(arc_from_back_to_start, color='r')
    >>> plt.show()
    """

    def __init__(self, start, end, center=None, points=np.linspace(0, 1, 100)):
        GreatArc.__init__(self, start, end, center=center, points=points)

        # Set the inner angle to be 2*pi radians, the full circle.
        self.inner_angle = 2 * np.pi * u.rad


