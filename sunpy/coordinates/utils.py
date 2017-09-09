#
# Calculates the co-ordinates along great arcs between two specified points
# which are assumed to be on disk.
#
from __future__ import absolute_import, division, print_function

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import frames

__all__ = ['GreatArc', 'GreatCircle']


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
        self._start_cartesian = self.start.transform_to(frames.Heliocentric).cartesian.xyz.to(self._distance_unit).value
        self._end_cartesian = self.end.transform_to(frames.Heliocentric).cartesian.xyz.to(self._distance_unit).value
        self._center_cartesian = self.center.transform_to(frames.Heliocentric).cartesian.xyz.to(self._distance_unit).value

        # Great arc properties calculation
        # Vector from center to first point
        self._v1 = self._start_cartesian - self._center_cartesian

        # Distance of the first point from the center
        self._r = np.linalg.norm(self._v1)

        # Vector from center to second point
        self._v2 = self._end_cartesian - self._center_cartesian

        # The v3 vector lies in plane of v1 & v2 and is orthogonal to v1
        self._v3 = np.cross(np.cross(self._v1, self._v2), self._v1)
        self._v3 = self._r * self._v3 / np.linalg.norm(self._v3)

        # Inner angle between v1 and v2 in radians
        self.inner_angle = np.arctan2(np.linalg.norm(np.cross(self._v1, self._v2)),
                                      np.dot(self._v1, self._v2)) * u.rad

        # Radius of the sphere
        self.radius = self._r * self._distance_unit

        # Distance on the sphere between the start point and the end point.
        self.distance = self.radius * self.inner_angle.value

        # Calculates the inner angles for the parameterized points along the arc
        # and returns the value in radians, from the start co-ordinate to the
        # end.
        self.inner_angles = self.points.reshape(self._npoints, 1)*self.inner_angle

        # Distances of the points along the great arc from the start to end
        # co-ordinate.  The units are defined as those returned after
        # transforming the co-ordinate system of the start co-ordinate into
        # its Cartesian equivalent.
        self.distances = self.radius * self.inner_angles.value

        # Co-ordinates along the great arc in the Cartesian co-ordinate frame.
        self._great_arc_points_cartesian = (self._v1[np.newaxis, :] * np.cos(self.inner_angles) +
                                            self._v3[np.newaxis, :] * np.sin(self.inner_angles) +
                                            self._center_cartesian) * self._distance_unit
        self.cartesian_coordinates = SkyCoord(self._great_arc_points_cartesian[:, 0],
                                              self._great_arc_points_cartesian[:, 1],
                                              self._great_arc_points_cartesian[:, 2],
                                              frame=frames.Heliocentric,
                                              observer=self._observer)

        # Co-ordinates along the great arc in the co-ordinate frame of the
        # start point.
        self.coordinates = self.cartesian_coordinates.transform_to(self._start_frame)

        # Boolean array indicating which coordinates are on the front of the
        # disk (True) or on the back (False).
        self.front_or_back = self.cartesian_coordinates.z.value > 0

        # Calculate the indices where the co-ordinates change from being on the
        # front of the disk to the back to the disk.
        self._fob = self.front_or_back.astype(np.int)
        self._change = self._fob[1:] - self._fob[0:-1]
        self.from_front_to_back_index = np.where(self._change == -1)[0][0]
        self.from_back_to_front_index = np.where(self._change == 1)[0][0]

        # Indices of arc on the front side
        self.front_arc_indices = np.concatenate((np.arange(self.from_back_to_front_index, self._npoints),
                                                 np.arange(0, self.from_front_to_back_index)))

        # Indices of arc on the back side
        self.back_arc_indices = np.arange(self.from_front_to_back_index + 1,
                                          self.from_back_to_front_index)

        # Calculate the exact location where the great circle intersects
        # moves over the edge of the disk

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


