# coding: utf-8
"""
=============================
Drawing and using a Great Arc
=============================

How to define and draw a great arc on an image of the
Sun, and to extract intensity values along that arc.
"""
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates.frames import Heliocentric, Helioprojective
from sunpy.coordinates.utils import GreatArc, ArcVisibility
from sunpy.data.sample import AIA_171_IMAGE

###############################################################################
# We start with the sample data
m = sunpy.map.Map(AIA_171_IMAGE)

###############################################################################
# Let's define the start and end coordinates of the arc.
start = SkyCoord(735 * u.arcsec, -471 * u.arcsec, frame=m.coordinate_frame)
end = SkyCoord(-100 * u.arcsec, 800 * u.arcsec, frame=m.coordinate_frame)

###############################################################################
# Create the great arc between the start and end points.
great_arc = GreatArc(start, end)

###############################################################################
# Plot the great arc on the Sun.
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot(axes=ax)
on_front = {"color": 'y', "linewidth": 5, "label": 'visible'}
initial = {"color": 'r', "label": 'initial'}
target = {"color": 'r', "label": 'target'}
ax.plot_coord(great_arc.coordinates, **on_front)
ax.plot_coord(start, 'x', **initial)
ax.plot_coord(end, 'o', **target)
plt.show()

###############################################################################
# Now we can calculate the nearest integer pixels of the data that correspond
# to the location of arc.
pixels = np.asarray(np.rint(m.world_to_pixel(great_arc.coordinates)), dtype=int)
x = pixels[0, :]
y = pixels[1, :]

###############################################################################
# Get the intensity along the arc from the start to the end point.
intensity_along_arc = m.data[y, x]

###############################################################################
# Define the angular location of each pixel along the arc from the start point
# to the end.
angles = great_arc.angles.to(u.deg)

###############################################################################
# Plot the intensity along the arc from the start to the end point.
fig, ax = plt.subplots()
ax.plot(angles, intensity_along_arc)
ax.set_xlabel('degrees of arc from start')
ax.set_ylabel('intensity')
ax.grid(linestyle='dotted')
plt.show()

###############################################################################
# The example above draws an arc along the inner angle directed from the start
# to the end coordinate.  The outer angle can also be used to define the arc.
great_arc = GreatArc(start, end, use_inner_angle_direction=False)
coordinates = great_arc.coordinates


arc_visibility = ArcVisibility(coordinates)
from_back_to_front = arc_visibility.from_back_to_front
from_front_to_back = arc_visibility.from_front_to_back
fig = plt.figure()
ax = plt.subplot(projection=m)
m.plot(axes=ax)
on_back = {"color": 'c', "linewidth": 5, "label": 'not visible', "linestyle": ":"}
ax.plot_coord(coordinates[0:from_front_to_back], **on_front)
ax.plot_coord(coordinates[from_front_to_back+1: from_back_to_front], **on_back)
ax.plot_coord(coordinates[from_back_to_front+1:], **on_front)
ax.plot_coord(start, 'x', **initial)
ax.plot_coord(end, 'o', **target)
plt.show()



###############################################################################
# Great circles can also be drawn using the GreatArc object.  The following
# example creates a great circle that passes through two points on the solar
# surface, the first point seen from AIA and the second as seen from STEREO A.
aia_image = '/Users/ireland/Data/jp2/20181127/2018_11_27__14_01_57_34__SDO_AIA_AIA_171.jp2'
stereo_image = '/Users/ireland/Data/jp2/20181127/2018_11_27__14_09_02_833__STEREO-A_SECCHI_EUVI_171.jp2'
aia_map = sunpy.map.Map(aia_image)
stereo_map = sunpy.map.Map(stereo_image)

aia = SkyCoord(500*u.arcsec, -320*u.arcsec, observer=aia_map.observer_coordinate, frame=Helioprojective)
stereo = SkyCoord(-600*u.arcsec, 420*u.arcsec, observer=stereo_map.observer_coordinate, frame=Helioprojective)

aia_gc = GreatArc(aia, stereo, points=1000, great_circle=True, use_inner_angle_direction=False)

# Visibility of the arc as seen from AIA.
aia_visibility = coordinates.visibility

# Visibility of the arc as seen from STEREO A.
stereo_visibility = coordinates.transform_to(stereo.frame).transform_to(Heliocentric).z.value > 0

# The part of the arc which is visible from AIA and STEREO A.
both = np.logical_and(aia_visibility.visibility, stereo_visibility.visibility)

# The part of the arc which is not visible from either AIA or STEREO A.
neither = np.logical_and(~aia_visibility.visibility, ~stereo_visibility.visibility)


###############################################################################
# Helper function to make a nice plot
def great_circle_visibility_helper(visibility):
    """
    Find a set of indices such that the corresponding coordinates don't
    flip from one side of the disk to another.

    This function assumes that True means that the corresponding
    coordinate is on the front side of the disk.
    """
    front = visibility.astype(np.int)
    change = front[1:] - front[0:-1]
    from_back_to_front = np.where(change == 1)[0][0]

    logic = np.roll(visibility, -from_back_to_front - 1)
    indices = np.roll(np.arange(0, len(visibility)), -from_back_to_front - 1)
    return indices[logic]


###############################################################################
# Determine the indices of the coordinates that are on and not on the
# observable disk of the Sun as seen from AIA and STEREO
aia_front_arc_indices = great_circle_visibility_helper(aia_visibility)
aia_back_arc_indices = great_circle_visibility_helper(~aia_visibility)

stereo_front_arc_indices = great_circle_visibility_helper(stereo_visibility)
stereo_back_arc_indices = great_circle_visibility_helper(~stereo_visibility)

###############################################################################
# Plot the great circle and its visibility on both the AIA and STEREO maps
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1, projection=aia_map)
aia_map.plot(axes=ax1)

###############################################################################
# Set up the colors and linestyles we want and create the plot.
visible_to_both = {"color": 'k', "label": 'visible to both'}
visible_to_neither = {"color": 'r', "label": 'visible to neither'}

ax1.plot_coord(coordinates[aia_front_arc_indices], **on_front)
ax1.plot_coord(coordinates[aia_back_arc_indices], **on_back)
ax1.plot_coord(coordinates[both], **visible_to_both)
ax1.plot_coord(coordinates[neither], **visible_to_neither)
ax1.plot_coord(aia, 'x', **initial)
ax1.plot_coord(stereo, 'o', **target)
plt.legend()

ax2 = fig.add_subplot(1, 2, 2, projection=stereo_map)
stereo_map.plot(axes=ax2)
ax2.plot_coord(coordinates.transform_to(stereo.frame)[stereo_front_arc_indices], **on_front)
ax2.plot_coord(coordinates.transform_to(stereo.frame)[stereo_back_arc_indices], **on_back)
ax2.plot_coord(coordinates.transform_to(stereo.frame)[both], **visible_to_both)
ax2.plot_coord(coordinates.transform_to(stereo.frame)[neither], **visible_to_neither)
ax2.plot_coord(aia.transform_to(stereo.frame), 'x', **initial)
ax2.plot_coord(stereo.transform_to(stereo.frame), 'o', **target)

plt.show()
