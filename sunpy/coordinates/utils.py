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

    points : `None`, `int`, `~numpy.ndarray`
        Number of points along the great arc.  If None, the arc is calculated
        at 100 equally spaced points from start to end.  If int, the arc is
        calculated at "points" equally spaced points from start to end.  If a
        numpy.ndarray is passed, it must be one dimensional and have values
        >=0 and <=1.  The values in this array correspond to parameterized
        locations along the great arc from zero, denoting the start of the arc,
        to 1, denoting the end of the arc.  Setting this keyword on initializing
        a GreatArc object sets the locations of the default points along the
        great arc.

    Methods
    -------
    inner_angles : `~astropy.units.rad`
        Radian angles of the points along the great arc from the start to end
        co-ordinate.

    distances : `~astropy.units`
        Distances of the points along the great arc from the start to end
        co-ordinate.  The units are defined as those returned after transforming
        the co-ordinate system of the start co-ordinate into its Cartesian
        equivalent.

    coordinates : `~astropy.coordinates.SkyCoord`
        Co-ordinates along the great arc in the co-ordinate frame of the
        start point.

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
    >>> ax.plot_coord(great_arc.coordinates(), color='c')
    >>> plt.show()

    """

    def __init__(self, start, end, center=None, points=None):
        # Start point of the great arc
        self.start = start

        # End point of the great arc
        self.end = end

        # Parameterized location of points between the start and the end of the
        # great arc.
        # Default parameterized points location.
        self.default_points = np.linspace(0, 1, 100)

        # If the user requests a different set of default parameterized points
        # on initiation of the object, then these become the default.  This
        # allows the user to use the methods without having to specify their
        # choice of points over and over again, while also allowing the
        # flexibility in the methods to calculate other values.
        self.default_points = self._points_handler(points)

        # Units of the start point
        self.distance_unit = self.start.transform_to(frames.Heliocentric).cartesian.xyz.unit

        # Co-ordinate frame
        self.start_frame = self.start.frame

        # Observer
        self.observer = self.start.observer

        # Set the center of the sphere
        if center is None:
            self.center = SkyCoord(0 * self.distance_unit,
                                   0 * self.distance_unit,
                                   0 * self.distance_unit, frame=frames.Heliocentric)

        # Convert the start, end and center points to their Cartesian values
        self.start_cartesian = self.start.transform_to(frames.Heliocentric).cartesian.xyz.to(self.distance_unit).value
        self.end_cartesian = self.end.transform_to(frames.Heliocentric).cartesian.xyz.to(self.distance_unit).value
        self.center_cartesian = self.center.transform_to(frames.Heliocentric).cartesian.xyz.to(self.distance_unit).value

        # Great arc properties calculation
        # Vector from center to first point
        self.v1 = self.start_cartesian - self.center_cartesian

        # Distance of the first point from the center
        self._r = np.linalg.norm(self.v1)

        # Vector from center to second point
        self.v2 = self.end_cartesian - self.center_cartesian

        # The v3 vector lies in plane of v1 & v2 and is orthogonal to v1
        self.v3 = np.cross(np.cross(self.v1, self.v2), self.v1)
        self.v3 = self._r * self.v3 / np.linalg.norm(self.v3)

        # Inner angle between v1 and v2 in radians
        self.inner_angle = np.arctan2(np.linalg.norm(np.cross(self.v1, self.v2)),
                                        np.dot(self.v1, self.v2)) * u.rad

        # Radius of the sphere
        self.radius = self._r * self.distance_unit

        # Distance on the sphere between the start point and the end point.
        self.distance = self.radius * self.inner_angle.value

    def _points_handler(self, points):
        """
        Interprets the points keyword.
        """
        if points is None:
            return self.default_points
        elif isinstance(points, int):
            return np.linspace(0, 1, points)
        elif isinstance(points, np.ndarray):
            if points.ndim > 1:
                raise ValueError('One dimensional numpy ndarrays only.')
            if np.any(points < 0) or np.any(points > 1):
                raise ValueError('All value in points array must be strictly >=0 and <=1.')
            return points
        else:
            raise ValueError('Incorrectly specified "points" keyword value.')

    def inner_angles(self, points=None):
        """
        Calculates the inner angles for the parameterized points along the arc
        and returns the value in radians, from the start co-ordinate to the
        end.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        inner_angles : `~astropy.units.rad`
            Radian angles of the points along the great arc from the start to
            end co-ordinate.

        """
        these_points = self._points_handler(points)
        return these_points.reshape(len(these_points), 1)*self.inner_angle

    def distances(self, points=None):
        """
        Calculates the distance from the start co-ordinate to the end
        co-ordinate on the sphere for all the parameterized points.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        distances : `~astropy.units`
            Distances of the points along the great arc from the start to end
            co-ordinate.  The units are defined as those returned after
            transforming the co-ordinate system of the start co-ordinate into
            its Cartesian equivalent.
        """
        return self.radius * self.inner_angles(points=points).value

    def coordinates(self, points=None):
        """
        Calculates the co-ordinates on the sphere from the start to the end
        co-ordinate for all the parameterized points.  Co-ordinates are
        returned in the frame of the start coordinate.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        arc : `~astropy.coordinates.SkyCoord`
            Co-ordinates along the great arc in the co-ordinate frame of the
            start point.

        """
        return self.cartesian_coordinates(points=points).transform_to(self.start_frame)

    def cartesian_coordinates(self, points=None):
        """
        Calculates the co-ordinates on the sphere from the start to the end
        co-ordinate for all the parameterized points.  Co-ordinates are
        returned in the Cartesian co-ordinate frame.

        Parameters
        ----------
        points : `None`, `int`, `~numpy.ndarray`
            If None, use the default locations of parameterized points along the
            arc.  If int, the arc is calculated at "points" equally spaced
            points from start to end.  If a numpy.ndarray is passed, it must be
            one dimensional and have values >=0 and <=1.  The values in this
            array correspond to parameterized locations along the great arc from
            zero, denoting the start of the arc, to 1, denoting the end of the
            arc.

        Returns
        -------
        arc : `~astropy.coordinates.SkyCoord`
            Co-ordinates along the great arc in the Cartesian co-ordinate frame.

        """
        # Calculate the inner angles
        these_inner_angles = self.inner_angles(points=points)

        # Calculate the Cartesian locations from the first to second points
        great_arc_points_cartesian = (self.v1[np.newaxis, :] * np.cos(these_inner_angles) +
                                      self.v3[np.newaxis, :] * np.sin(these_inner_angles) +
                                      self.center_cartesian) * self.distance_unit

        # Return the coordinates of the great arc between the start and end
        # points
        return SkyCoord(great_arc_points_cartesian[:, 0],
                        great_arc_points_cartesian[:, 1],
                        great_arc_points_cartesian[:, 2],
                        frame=frames.Heliocentric,
                        observer=self.observer)


class GreatCircle(GreatArc):
    def __init__(self, start, end, center=None, points=None):
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

        points : `None`, `int`, `~numpy.ndarray`
            Number of points along the great arc.  If None, the arc is calculated
            at 100 equally spaced points from start to end.  If int, the arc is
            calculated at "points" equally spaced points from start to end.  If a
            numpy.ndarray is passed, it must be one dimensional and have values
            >=0 and <=1.  The values in this array correspond to parameterized
            locations along the great arc from zero, denoting the start of the arc,
            to 1, denoting the end of the arc.  Setting this keyword on initializing
            a GreatArc object sets the locations of the default points along the
            great arc.

        Methods
        -------
        inner_angles : `~astropy.units.rad`
            Radian angles of the points along the great arc from the start to end
            co-ordinate.

        distances : `~astropy.units`
            Distances of the points along the great arc from the start to end
            co-ordinate.  The units are defined as those returned after transforming
            the co-ordinate system of the start co-ordinate into its Cartesian
            equivalent.

        coordinates : `~astropy.coordinates.SkyCoord`
            Co-ordinates along the great arc in the co-ordinate frame of the
            start point.

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
        >>> great_circle = GreatCircle(a, b)
        >>> coordinates = great_circle.coordinates(points=1000)
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
        GreatArc.__init__(self, start, end, center=center, points=points)

        # Set the inner angle to be 2*pi radians, the full circle.
        self.inner_angle = 2 * np.pi * u.rad

        # Boolean array indicating which coordinate is on the front of the disk
        # (True) or on the back (False).
        self.front_or_back = self.cartesian_coordinates().z.value > 0

        # Calculate the indices where the co-ordinates change from being on the
        # front of the disk to the back to the disk.
        self._fob = self.front_or_back.astype(np.int)
        self._change = self._fob[1:] - self._fob[0:-1]
        self.from_front_to_back_index = np.where(self._change == -1)[0][0]
        self.from_back_to_front_index = np.where(self._change == 1)[0][0]

        # Indices of arcs on the front side and the back
        self.front_arc_indices = np.concatenate((np.arange(self.from_back_to_front_index, len(self.coordinates())),
                                                 np.arange(0, self.from_front_to_back_index)))

        self.back_arc_indices = np.arange(self.from_front_to_back_index + 1,
                                          self.from_back_to_front_index)
