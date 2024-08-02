"""PV class."""

import epics as _epics

from ... import _get_facility


class PV:
    """PV class."""

    def __init__(
        self,
        devname,
        propty,
        connection_timeout=None
    ):
        """."""
        self.devname = devname
        self.propty = propty
        facil = _get_facility()
        pvname = facil.get_attribute_from_aliases('cs_devname', devname)
        pvname += facil.get_attribute_from_aliases(
            f'cs_propties.{propty}.name', devname
        )

        self._pvo = _epics.PV(
            pvname, connection_timeout=connection_timeout, auto_monitor=False
        )

    @property
    def connected(self):
        """."""
        return self._pvo.connected

    @property
    def value(self):
        """."""
        return self.get()

    @value.setter
    def value(self, value):
        """."""
        self.put(value, wait=False)

    @property
    def timestamp(self):
        """."""
        return self._pvo.timestamp

    @property
    def host(self):
        """."""
        return self._pvo.host

    @property
    def units(self):
        """."""
        return self._pvo.units

    @property
    def lower_limit(self):
        """."""
        return self._pvo.lower_limit

    @property
    def upper_limit(self):
        """."""
        return self._pvo.upper_limit

    def get(self, timeout=None):
        """."""
        return self._pvo.get(timeout=timeout)

    def put(self, value, wait=False):
        """."""
        return self._pvo.put(value, wait=wait)

    def wait_for_connection(self, timeout=None):
        """."""
        return self._pvo.wait_for_connection(timeout=timeout)
