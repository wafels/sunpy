# -*- coding: utf-8 -*-
# Author: Florian Mayer <florian.mayer@bitsrc.org>
#
# This module was developed with funding provided by
# the ESA Summer of Code (2011).
#
# pylint: disable=C0103,R0903

""" Facilities to interface with the HEK. """

from __future__ import absolute_import

import json
import codecs

from itertools import chain
from datetime import datetime
from sunpy.net import attr
from sunpy.net.hek import attrs
from sunpy.net.vso import attrs as v_attrs
from sunpy.util import unique
from sunpy.util.xml import xml_to_dict
from sunpy.extern.six import iteritems
from sunpy.extern.six.moves import urllib
from sunpy.util import deprecated
from sunpy.time import Time as apTime

__all__ = ['HEKClient']

DEFAULT_URL = 'http://www.lmsal.com/hek/her'


def _freeze(obj):
    """ Create hashable representation of result dict. """
    if isinstance(obj, dict):
        return tuple((k, _freeze(v)) for k, v in iteritems(obj))
    if isinstance(obj, list):
        return tuple(_freeze(elem) for elem in obj)
    return obj


class HEKClient(object):
    """ Client to interact with the Heliophysics Event Knowledgebase (HEK).
    The HEK stores solar feature and event data generated by algorithms and
    human observers."""
    # FIXME: Expose fields in .attrs with the right types
    # that is, not all StringParamWrapper!

    default = {
        'cosec': '2',
        'cmd': 'search',
        'type': 'column',
        'event_type': '**',
    }
    # Default to full disk.
    attrs.walker.apply(attrs.SpatialRegion(), {}, default)

    def __init__(self, url=DEFAULT_URL):
        self.url = url

    def _download(self, data):
        """ Download all data, even if paginated. """
        page = 1
        results = []
        reader = codecs.getreader("utf-8")

        while True:
            data['page'] = page
            fd = urllib.request.urlopen(
                self.url, urllib.parse.urlencode(data).encode('utf-8'))
            try:
                result = json.load(reader(fd))
            finally:
                fd.close()
            results.extend(result['result'])

            if not result['overmax']:
                return list(map(Response, results))
            page += 1

    def search(self, *query):
        """ Retrieves information about HEK records matching the criteria
        given in the query expression. If multiple arguments are passed,
        they are connected with AND. The result of a query is a list of
        unique HEK Response objects that fulfill the criteria."""
        query = attr.and_(*query)

        data = attrs.walker.create(query, {})
        ndata = []
        for elem in data:
            new = self.default.copy()
            new.update(elem)
            ndata.append(new)

        if len(ndata) == 1:
            return self._download(ndata[0])
        else:
            return self._merge(self._download(data) for data in ndata)

    @deprecated('0.8', alternative='HEKClient.search')
    def query(self, *query):
        """
        See `~sunpy.net.hek.hek.HEKClient.fetch`
        """
        return self.search(*query)


    def _merge(self, responses):
        """ Merge responses, removing duplicates. """
        return list(unique(chain.from_iterable(responses), _freeze))


class Response(dict):
    """Handles the response from the HEK.  Each Response object is a subclass
    of the dictionary object.  The dictionary key-value pairs correspond to the
    HEK feature/event properties and their values, for that record from the
    HEK.  Each Response object also has extra properties that relate HEK
    concepts to VSO concepts."""
    @property
    def vso_time(self):
        return v_attrs.Time(
            apTime.strptime(self['event_starttime'], "%Y-%m-%dT%H:%M:%S"),
            apTime.strptime(self['event_endtime'], "%Y-%m-%dT%H:%M:%S")
        )

    @property
    def vso_instrument(self):
        if self['obs_instrument'] == 'HEK':
            raise ValueError("No instrument contained.")
        return v_attrs.Instrument(self['obs_instrument'])

    @property
    def vso_all(self):
        return attr.and_(self.vso_time, self.vso_instrument)

    def get_voevent(self, as_dict=True,
                    base_url="http://www.lmsal.com/hek/her?"):
        """Retrieves the VOEvent object associated with a given event and
        returns it as either a Python dictionary or an XML string."""

        # Build URL
        params = {
            "cmd": "export-voevent",
            "cosec": 1,
            "ivorn": self['kb_archivid']
        }
        url = base_url + urllib.parse.urlencode(params)

        # Query and read response
        response = urllib.request.urlopen(url).read()

        # Return a string or dict
        if as_dict:
            return xml_to_dict(response)
        else:
            return response
