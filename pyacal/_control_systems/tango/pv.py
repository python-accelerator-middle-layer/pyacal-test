"""Sirius PV class."""

import tango as _tango


from ... import _get_connections_dict


# NOTE: I know nothing about tango. This class must be implemented.
class PV(_tango.DeviceProxy):
    """."""

    def __init__(self, devname, propty, **kwargs):
        """."""
        self.devname = devname
        self.propty = propty
        super().__init__(devname, propty, **kwargs)  # NOTE: not sure...
        _get_connections_dict()[(devname, propty)] = self

    @property
    def connected(self):
        """."""
        raise NotImplementedError('Please Implement me.')

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
