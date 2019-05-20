"""
=======================================
Applying Differential Rotation to a Map
=======================================

Often it is useful to use an image of the Sun to get an estimate of what the
Sun looks like at some earlier or later time. Doing this requires two pieces
of information, the solar differential rotation rate, and the location of the
observer at the earlier/later time. This example gives a short introduction to
of how to do this.
"""

##############################################################################
# Start by importing the necessary modules.
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import TimeDelta

import sunpy.map
import sunpy.data.sample
from sunpy.physics.differential_rotation import differential_rotate

##############################################################################
# Load in an AIA map:
aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

##############################################################################
# Differentially rotate the AIA map.  Note that the location of the observer
# at the later time has not been explicitly specified.  When the observer has
# not been explicitly specified, it is assumed that the observer is on the
# Earth.  Hence the commands below differentially rotates the image by four days
# and shifts the observer to the location of the Earth four days in the future.
dmap_time = differential_rotate(aia_map, time=4*u.day)

# Now let's plot the differentially rotated map
fig = plt.figure()
ax = plt.subplot(projection=dmap_time)
dmap_time.plot()
ax.set_title('The effect of 4 days of differential rotation')
dmap_time.draw_grid()


##############################################################################
# The command below does specify a new observer.  In this case the location of
# the observer is the Earth 5 days earlier
dmap_observer = differential_rotate(aia_map, observer=get_earth(aia_map.date - 5*u.day))

# Now let's plot the differentially rotated map
fig = plt.figure()
ax = plt.subplot(projection=dmap_observer)
dmap_observer.plot()
ax.set_title('The effect of 4 days of differential rotation')
dmap_observer.draw_grid()
