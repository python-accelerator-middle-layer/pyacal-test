"""Sirius PV class."""

from ... import _get_connections_dict, _get_facility, get_alias_from_devname, _get_simulator


class PV:
    """."""

    def __init__(self, devname, propty, **kwargs):
        """."""
        self.devname = devname
        self.propty = propty
        self.alias = get_alias_from_devname(devname)[propty]
        _get_connections_dict()[(devname, propty)] = self

    @property
    def connected(self):
        """."""
        return True

    @property
    def value(self):
        """."""
        get_from_simulator()

    @value.setter
    def value(self, value):
        raise NotImplementedError('Please Implement me.')

    def put(self, value, wait=False):
        """."""
        raise NotImplementedError('Please Implement me.')

    def wait_for_connection(self, timeout=None):
        """."""
        raise NotImplementedError('Please Implement me.')
