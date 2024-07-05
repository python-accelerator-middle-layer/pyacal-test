"""PV class."""

import tango as _tango

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
        cs_devname = facil.get_attribute_from_aliases('cs_devname', devname)
        cs_propty = facil.get_attribute_from_aliases(
            f'cs_propties.{propty}.name', devname
        )
        raise NotImplementedError('Please Implement me.')

    @property
    def connected(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    @property
    def auto_monitor(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    @auto_monitor.setter
    def auto_monitor(self, value):
        """."""
        raise NotImplementedError('Please Implement me.')

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
        raise NotImplementedError('Please Implement me.')

    @property
    def host(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    @property
    def units(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    @property
    def precision(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    @property
    def lower_limit(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    @property
    def upper_limit(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    def get(self, timeout=None):
        """."""
        raise NotImplementedError('Please Implement me.')

    def put(self, value, wait=False):
        """."""
        raise NotImplementedError('Please Implement me.')

    def wait_for_connection(self, timeout=None):
        """."""
        raise NotImplementedError('Please Implement me.')
