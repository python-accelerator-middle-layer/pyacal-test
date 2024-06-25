"""Sirius PV class."""

from ... import ALIAS_MAP


class PV():
    """."""

    def __init__(self, devname, property, **kwargs):
        """."""
        if devname not in ALIAS_MAP:
            raise ValueError(f'Devname {devname} does not exist.')
        elif property not in ALIAS_MAP[devname]:
            raise ValueError(f'Property {property} not valid.')

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
