"""Sirius PV class."""


import tango as _tango


# NOTE: I know nothing about tango. This class must be implemented.
class PV(_tango.DeviceProxy):
    """."""

    def __init__(self, *args, **kwargs):
        """."""
        super().__init__(*args, **kwargs)

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
