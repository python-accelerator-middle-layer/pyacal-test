"""Sirius PV class."""

import epics as _epics

from ... import FACILITY

ALL_CONNECTIONS = {}


class PV(_epics.pv.PV):
    """PV class."""

    def __init__(self, devname, propty, **kwargs):
        """."""
        self.devname = devname
        self.propty = propty
        mapping = FACILITY.alias_map[devname]
        pvname = mapping['cs_devname']
        pvname += mapping['cs_propties'][propty]['name']
        super().__init__(pvname, **kwargs)
        ALL_CONNECTIONS[(devname, propty)] = self
