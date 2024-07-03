"""PV class."""

import epics as _epics

from ... import _get_facility


class PV:
    """PV class."""

    def __init__(
        self,
        devname,
        propty,
        auto_monitor=True,
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
            pvname,
            auto_monitor=auto_monitor,
            connection_timeout=connection_timeout
        )

    @property
    def connected(self):
        """."""
        return self._pvo.connected

    @property
    def auto_monitor(self):
        """."""
        return self._pvo.auto_monitor

    @auto_monitor.setter
    def auto_monitor(self, value):
        """."""
        self._pvo.auto_monitor = int(value)

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
    def precision(self):
        """."""
        return self._pvo.precision

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
