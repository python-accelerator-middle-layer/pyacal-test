"""Sirius PV class."""

from ... import _get_connections_dict, _get_facility


class PV:
    """."""

    def __init__(self, devname, propty, **kwargs):
        """."""
        self.devname = devname
        self.propty = propty
        _get_connections_dict()[(devname, propty)] = self

    @property
    def connected(self):
        """."""
        return True

    @property
    def value(self):
        """."""
        raise NotImplementedError('Please Implement me.')

    @value.setter
    def value(self, value):
        raise NotImplementedError('Please Implement me.')

    def put(self, value, wait=False):
        """."""
        raise NotImplementedError('Please Implement me.')

    def wait_for_connection(self, timeout=None):
        """."""
        raise NotImplementedError('Please Implement me.')
