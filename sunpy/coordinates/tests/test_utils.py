import pytest

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

from sunpy.coordinates import sun
import sunpy.map as smap
import sunpy.data.test as test
from sunpy.coordinates import frames
from sunpy.coordinates.utils import GreatArc


@pytest.fixture
def aia171_test_map():
    return smap.Map(test.get_test_filepath('aia_171_level1.fits'))

# Test the great arc code against calculable quantities
# The inner angle is the same between each pair of co-ordinates.  You can
# calculate these co-ordinates using the inner angle formulae as listed here:
# https://en.wikipedia.org/wiki/Great-circle_distance


@pytest.mark.parametrize("initial, target", [((0, 0), (0, 45)),
                                        ((0, 0), (45, 0)),
                                        ((0, 45), (0, 0)),
                                        ((45, 0), (0, 0)),
                                        ((12, 13), (12, 58)),
                                        ((-10, 6), (-10, 51)),
                                        ((-20, -50), (-20, -5)),
                                        ((10, -50), (87.53163324626676, -55))])
def test_great_arc_calculable(initial, target):
    c = SkyCoord(initial[0]*u.degree, initial[1]*u.degree, frame=frames.HeliographicStonyhurst,
                 observer=frames.HeliographicStonyhurst(0*u.deg, 0*u.deg, 1*u.AU))
    d = SkyCoord(target[0]*u.degree, target[1]*u.degree, frame=frames.HeliographicStonyhurst,
                 observer=frames.HeliographicStonyhurst(0*u.deg, 0*u.deg, 1*u.AU)).transform_to(frames.Helioprojective)
    gc = GreatArc(c, d)

    c_trans = c.transform_to(frames.Heliocentric)
    assert gc._initial.x == c_trans.x
    assert gc._initial.y == c_trans.y
    assert gc._initial.z == c_trans.z
    assert gc._initial.observer.lat == 0*u.deg
    assert gc._initial.observer.lon == 0*u.deg
    assert gc._initial.observer.radius == 1 * u.AU

    d_trans = d.transform_to(frames.Heliocentric(observer=c.observer))
    np.testing.assert_almost_equal(gc._target.x.value, d_trans.x.value)
    np.testing.assert_almost_equal(gc._target.y.value, d_trans.y.value)
    np.testing.assert_almost_equal(gc._target.z.value, d_trans.z.value)
    assert gc._target.observer.lat == 0*u.deg
    assert gc._target.observer.lon == 0*u.deg
    assert gc._target.observer.radius == 1 * u.AU

    np.testing.assert_almost_equal(gc._angle.to('deg').value, 45.0)
    np.testing.assert_almost_equal(gc._radius.to('km').value, sun.constants.radius.to('km').value)
    np.testing.assert_almost_equal(gc._distance.to(
        'km').value, sun.constants.radius.to('km').value * 2 * np.pi/8, decimal=1)

    # Test that the initial and target coordinates stored by the great arc object
    # are the same as those passed in to the function
    assert gc.initial.lat == c.lat
    assert gc.initial.lon == c.lon
    assert gc.initial.radius == c.radius
    assert gc.initial.observer.lat == 0*u.deg
    assert gc.initial.observer.lon == 0*u.deg
    assert gc.initial.observer.radius == 1 * u.AU
    assert isinstance(gc.initial.frame, frames.HeliographicStonyhurst)

    assert gc.target.Tx == d.Tx
    assert gc.target.Ty == d.Ty
    assert gc.target.distance == d.distance
    assert gc.target.observer.lat == 0*u.deg
    assert gc.target.observer.lon == 0*u.deg
    assert gc.target.observer.radius == 1 * u.AU
    assert isinstance(gc.target.frame, frames.Helioprojective)


# Test the calculation of coordinates using varying numbers of points on
# initialization of the GreatArc object.
@pytest.mark.parametrize("points_requested, points_expected, first_point, last_point, last_inner_angle, last_distance",
                         # Test default
                         [(None, 100, (600, -600), (-100, 800), 1.8683580432741789, 1300377.1981299),
                          # Test int as an option
                          (3, 3, (600, -600), (-100, 800), 1.8683580432741789, 1300377.1981299),
                          # Test equally spaced monotonically increasing numpy
                          # array
                          (np.linspace(0, 1, 43), 43, (600, -600),
                           (-100, 800), 1.8683580432741789, 1300377.1981299),
                          # Test unequally spaced monotonically increasing numpy
                          # array
                          (np.asarray([0.1, 0.2, 0.6, 0.67, 0.99]), 5, (604.68091703, -468.64217597),
                           (-88.83212616, 792.76284375), 1.84967446, 1287373.4261486),
                          # Test unequally spaced monotonically decreasing numpy
                          # array
                          (np.asarray([0.93, 0.78, 0.3, 0.001]), 4, (-21.28208654, 743.58866798),
                           (600.1512768, -598.78376614), 0.00186836, 1300.37719813),
                          # Test numpy array that increases and decreases
                          (np.asarray([0.94, 0.73, 0.8, 0.21]), 4, (-32.5852606, 752.45507707),
                           (585.45829119, -305.26965043), 0.39235519, 273079.2116073)])
def test_great_arc_coordinates(points_requested, points_expected, first_point,
                               last_point, last_inner_angle, last_distance, aia171_test_map):
    coordinate_frame = aia171_test_map.coordinate_frame
    a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=coordinate_frame)
    b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=coordinate_frame)
    gc = GreatArc(a, b, points=points_requested)
    coordinates = gc.coordinates()
    angles = gc.angles()
    distances = gc.distances()

    # Ensure a GreatArc object is returned
    assert isinstance(gc, GreatArc)

    # Test the properties of the GreatArc object
    a_trans = a.transform_to(frames.Heliocentric)
    assert gc._initial.x == a_trans.x
    assert gc._initial.y == a_trans.y
    assert gc._initial.z == a_trans.z
    b_trans = b.transform_to(frames.Heliocentric(observer=a.observer))
    assert gc._target.x == b_trans.x
    assert gc._target.y == b_trans.y
    assert gc._target.z == b_trans.z
    assert gc._distance_unit == u.m
    assert gc._output_observer == a.observer
    assert gc._center.x == 0 * u.m
    assert gc._center.y == 0 * u.m
    assert gc._center.z == 0 * u.m

    assert u.allclose(gc._initial_cartesian * u.m, np.asarray(
        [428721.0913539, -428722.9051924, 341776.0910214]) * u.km)
    assert u.allclose(gc._target_cartesian * u.m, np.asarray(
        [-71429.5229381, 571439.071248, 390859.5797815]) * u.km)
    assert u.allclose(gc._center_cartesian * u.m, np.asarray([0, 0, 0]) * u.km)

    assert u.allclose(gc._v1 * u.m, np.asarray(
        [428721.0913539, -428722.9051924, 341776.0910214]) * u.km)
    assert u.allclose(gc._r, 696000000.0015007)
    assert u.allclose(gc._v2 * u.m, np.asarray(
        [-71429.5229381, 571439.071248, 390859.5797815]) * u.km)
    assert u.allclose(gc._v3 * u.m, np.asarray(
        [56761.6265851, 466230.7005856, 513637.0815867]) * u.km)

    # Inner angle
    assert gc._angle.unit == u.rad
    np.testing.assert_almost_equal(gc._angle.value, 1.8683580432741789)

    # Distance
    assert gc._distance.unit == u.m
    np.testing.assert_approx_equal(gc._distance.value, 1300377198.1299164)

    # Radius of the sphere
    assert gc._radius.unit == u.m
    assert u.isclose(gc._radius.value * u.m, 696000.000001501 * u.km)

    # Test the calculation of the SkyCoords
    # Coordinates method
    # Number of points
    assert len(coordinates) == points_expected

    # initial and target coordinates
    np.testing.assert_almost_equal(coordinates[0].Tx.value, first_point[0])
    np.testing.assert_almost_equal(coordinates[0].Ty.value, first_point[1])
    np.testing.assert_almost_equal(coordinates[-1].Tx.value, last_point[0])
    np.testing.assert_almost_equal(coordinates[-1].Ty.value, last_point[1])

    # Inner angles method
    # Inner angles
    assert len(angles) == points_expected
    np.testing.assert_almost_equal(angles[-1].value, last_inner_angle)

    # Distances method
    assert len(distances) == points_expected
    assert u.isclose(distances[-1].value * u.m, last_distance * u.km)


# Test that the great arc code rejects wrongly formatted points
@pytest.mark.parametrize("points", [np.asarray([[0, 0.1], [0.2, 0.3]]),
                                    np.asarray([0.1, 0.2, -0.1, 0.4]),
                                    np.asarray([0.3, 1.1, 0.6, 0.7]),
                                    'strings_not_permitted'])
def test_great_arc_wrongly_formatted_points(points, aia171_test_map):
    coordinate_frame = aia171_test_map.coordinate_frame
    a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=coordinate_frame)
    b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=coordinate_frame)
    with pytest.raises(ValueError):
        GreatArc(a, b, points=points)

    with pytest.raises(ValueError):
        GreatArc(a, b).coordinates(points=points)

    with pytest.raises(ValueError):
        GreatArc(a, b).angles(points=points)

    with pytest.raises(ValueError):
        GreatArc(a, b).distances(points=points)

    with pytest.raises(ValueError):
        GreatArc(a, b).distances(points=points)


# Test that the great arc code properly differentiates between the default
# points and the requested points
def test_great_arc_points_differentiates(aia171_test_map):
    coordinate_frame = aia171_test_map.coordinate_frame
    a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=coordinate_frame)
    b = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=coordinate_frame)
    gc = GreatArc(a, b)
    coordinates = gc.coordinates(10)
    angles = gc.angles(11)
    distances = gc.distances(12)
    assert len(coordinates) == 10 and len(gc.coordinates()) == 100
    assert len(angles) == 11 and len(gc.angles()) == 100
    assert len(distances) == 12 and len(gc.distances()) == 100


# Test that the great arc code properly understands different observers
# for the initial and target points
def test_great_arc_different_observer(aia171_test_map):
    a = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=aia171_test_map.coordinate_frame)

    observer = SkyCoord(-10.0*u.deg, 83*u.deg, radius=0.9*u.au,
                        frame=frames.HeliographicStonyhurst, obstime=aia171_test_map.date)
    b = SkyCoord(400*u.arcsec, 600*u.arcsec, observer=observer, frame=frames.Helioprojective)

    # Test that the input observers are indeed different
    assert a.observer.lon != b.observer.lon
    assert a.observer.lat != b.observer.lat
    assert a.observer.radius != b.observer.radius

    # Create the great arc
    gc = GreatArc(a, b)

    # The initial and target points stored internally are Heliocentric
    initial = gc._initial
    assert isinstance(initial.frame, frames.Heliocentric)
    target = gc._target
    assert isinstance(target.frame, frames.Heliocentric)

    # The initial and target points stored internally have the same observer
    assert initial.observer.lon == target.observer.lon
    assert initial.observer.lat == target.observer.lat
    assert initial.observer.radius == target.observer.radius

    # The initial point stored internally has the Heliocentric coordinates of the initial coordinate passed in.
    a2h = a.transform_to(frames.Heliocentric)
    assert initial.x == a2h.x
    assert initial.y == a2h.y
    assert initial.z == a2h.z

    # The target point stored internally has the Heliocentric coordinates of the initial coordinate passed in.
    b2h = b.transform_to(frames.Heliocentric(observer=aia171_test_map.observer_coordinate))

    # Missing an dp on b2h compared to target (TODO BUG?)
    np.testing.assert_almost_equal(target.x.value, b2h.x.value)
    np.testing.assert_almost_equal(target.y.value, b2h.y.value)
    np.testing.assert_almost_equal(target.z.value, b2h.z.value)


# Test that the great arc code properly decides between calculating
# a great arc and a great circle, and the direction of the arc.
def test_great_arc_great_circle_and_directions(aia171_test_map):
    a = SkyCoord(700*u.arcsec, -100*u.arcsec, frame=aia171_test_map.coordinate_frame)

    observer = SkyCoord(25.0*u.deg, -9*u.deg, radius=1.5*u.au,
                        frame=frames.HeliographicStonyhurst, obstime=aia171_test_map.date)
    b = SkyCoord(-400*u.arcsec, -600*u.arcsec, observer=observer, frame=frames.Helioprojective)

    # Create the great arc and test the defaults
    gc = GreatArc(a, b)
    assert gc.great_circle is False
    assert gc.use_inner_angle_direction is True
    inner_angle = gc._angle

    # Test great circle selection
    gc = GreatArc(a, b, great_circle=True)
    assert gc.great_circle is True
    np.testing.assert_almost_equal(gc._angle.value, 2*np.pi)
    assert gc._angle.unit == u.rad

    # Test outer angle selection
    gc = GreatArc(a, b, use_inner_angle_direction=False)
    assert gc.use_inner_angle_direction is False
    np.testing.assert_almost_equal(gc._angle.value, inner_angle.value - 2*np.pi)
    assert gc._angle.unit == u.rad

    # Test great circle selection and outer angle selection simultaneously
    gc = GreatArc(a, b, great_circle=True, use_inner_angle_direction=False)
    assert gc.great_circle is True
    assert gc.use_inner_angle_direction is False
    np.testing.assert_almost_equal(gc._angle.value, -2*np.pi)
    assert gc._angle.unit == u.rad

