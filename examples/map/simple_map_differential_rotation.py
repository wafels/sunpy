"""
=======================================
Applying Differential Rotation to a Map
=======================================

Often it is useful to use an image of the Sun to get an estimate of what the
Sun looks like at some earlier or later time. Doing this requires two pieces
of information, the solar differential rotation rate, and the location of the
observer at the earlier/later time. This example gives a short introduction to
of how to do this using the differential_rotate function.
"""

##############################################################################
# Start by importing the necessary modules.
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, get_body
from astropy.time import TimeDelta

import sunpy.map
import sunpy.data.sample
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.coordinates.frames import HeliographicStonyhurst

##############################################################################
# Load in an AIA map:
aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

##############################################################################
# The amount of differential rotation to apply is calculated using the
# difference between the map observation time and new observation time.  The
# new observation time can be specified using a duration or a new date.  For
# example, to rotate a map four days in to the future you can
#
dmap_time = differential_rotate(aia_map, time=4*u.day)

# Equivalently, the time keyword can also accept a TimeDelta object or a
# sunpy-compatible date/time.  This specifies the amount of differential
# rotation to apply to each pixel.  Use of the time keyword also assumes
# that the observer at the new observation time is at the Earth.  This means that
# after the solar differential rotation is calculated for each pixel, the data
# is then transformed to take into account the change in the point of
# view due to the new observer location.


# Now let's plot the differentially rotated map
fig = plt.figure()
ax = plt.subplot(projection=dmap_time)
dmap_time.plot()
ax.set_title('Differentially rotated 4 days in to the futture')
dmap_time.draw_grid()


##############################################################################
# The command below does specify a new observer.  In this case the location of
# the observer is the Earth 5 days earlier
dmap_observer = differential_rotate(aia_map, observer=get_body('earth', aia_map.date - 5*u.day))

# Now let's plot the differentially rotated map
fig = plt.figure()
ax = plt.subplot(projection=dmap_observer)
dmap_observer.plot()
ax.set_title('Differentially rotated 5 days in to the past')
dmap_observer.draw_grid()
