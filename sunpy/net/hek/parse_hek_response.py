#
# Parse the HEK response dictionaries, and return quantities, SkyCoords, etc.
# This turns HEK response in to something useful for the rest of SunPy.
#
from astropy.coordinates import SkyCoord
import astropy.units as u

from sunpy.coordinates import frames
from sunpy.time import parse_time

#
# Functions that convert the string values of the dictionary keys in to the
# relevant SunPy objects.
#
def hpc_boundcc(hek_dictionary, event_measurement_time='event_starttime'):
    """
    Parses the Helioprojective Cartesian boundary coordinates and returns
    SkyCoords.
    """
    p1 = hek_dictionary["hpc_boundcc"][9: -2]
    p2 = p1.split(',')
    p3 = [v.split(" ") for v in p2]
    date = parse_time(hek_dictionary[event_measurement_time])
    return SkyCoord([(float(v[0]), float(v[1]))*u.arcsec for v in p3], obstime=date, frame=frames.Helioprojective)

def parse_hek_response(hek_response):
    # Which feature/event is this?

    # Determine the required properties

    # Parse the required properties and update the response dictionary

    # Determine the optional properties

    # Parse the optional properties and update the response dictionary

    return hek_response
