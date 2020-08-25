"""SDO Map subclass definitions"""

import numpy as np

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, HeliocentricMeanEcliptic
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from sunpy.map import GenericMap
from sunpy.map.sources.source_type import source_stretch

__all__ = ['EOVSAMap']


class EOVSAMap(GenericMap):
    """EOVSA Image Map.


    Notes
    -----

    References
    ----------
    """

    def __init__(self, data, header, **kwargs):
        super().__init__(data, header, **kwargs)

        # Fill in some missing info
        self.meta['detector'] = self.meta.get('detector', "EOVSA")
        self._nickname = self.detector
        self.plot_settings['cmap'] = None # self._get_cmap_name()
        self.plot_settings['norm'] = ImageNormalize(
            stretch=source_stretch(self.meta, AsinhStretch(0.01)), clip=False)

    @property
    def observatory(self):
        """
        Returns the observatory.
        """
        return self.meta.get('telescop', '')

    @property
    def measurement(self):
        """
        Returns the measurement in more commonly used units
        """
        return self.wavelength.to(u.GHz)

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Determines if header corresponds to an EOVSA image"""
        return str(header.get('telescop', '')).startswith('EOVSA')

