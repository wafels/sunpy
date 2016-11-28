"""Temperature Map subclass definitions"""
from __future__ import absolute_import, division, absolute_import
#pylint: disable=W0221,W0222,E1101,E1121

__author__ = "Jack Ireland"
__email__ = "jack.ireland@nasa.gov"

from sunpy.map import GenericMap
import matplotlib.cm as cm
from matplotlib import colors

__all__ = ['TemperatureMap']


class TemperatureMap(GenericMap):
    """Temperature Image Map

    A map class for temperature maps that have been written out to FITS
    files and read in.

    """

    def __init__(self, data, header, **kwargs):

        GenericMap.__init__(self, data, header, **kwargs)

        # There is no detector for the temperature maps, so why not
        # replace it with an indication of the algorithm used to
        # determine the temperature.  Note that this relies on the
        # FITS files having an indication that the file contains a
        # temperature map
        self.meta['detector'] = 'HK2012'
        self.meta['obsrvtry'] = None
        self._nickname = self.detector
        # Colour maps
        self.plot_settings['cmap'] = cm.viridis
        self.plot_settings['norm'] = colors.Normalize()

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Determines if header corresponds to a temperature image."""
        return header.get('ORIGIN') == 'TEMPERATURE'

    @property
    def measurement(self):
        """
        Returns the measurement type.
        """
        return 'temperature'
