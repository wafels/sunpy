"""
================
AIA Plot Example
================

This is a very simple way to plot a sample AIA image.
"""
import matplotlib.pyplot as plt

import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE

###############################################################################
# We now create the Map using the sample data.

aiamap = sunpy.map.Map(AIA_171_IMAGE)

###############################################################################
# Now we do a quick plot.

fig = plt.figure()
aiamap.plot()
plt.show()
