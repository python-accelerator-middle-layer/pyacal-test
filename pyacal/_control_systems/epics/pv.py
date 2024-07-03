"""Sirius PV class."""

import epics as _epics

from ... import _get_connections_dict, _get_facility


class PV(_epics.pv.PV):
    """PV class."""

    def __init__(self, devname, propty, **kwargs):
        """."""
        self.devname = devname
        self.propty = propty
        facil = _get_facility()
        pvname = facil.get_attribute_from_aliases('cs_devname', devname)
        pvname += facil.get_attribute_from_aliases(
            f'cs_propties.{propty}.name', devname
        )
        super().__init__(pvname, **kwargs)
        _get_connections_dict()[(devname, propty)] = self
