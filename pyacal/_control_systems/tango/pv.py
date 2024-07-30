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
        super().__init__(devname, **kwargs)
        _get_connections_dict()[(devname, propty)] = self

    @property
    def connected(self):
        try:
            self.state()
            return True
        except _tango.DevFailed as e:
            print(e)
            return False

    @property
    def value(self):
        return self.read_attribute(self.propty)

    @value.setter
    def value(self, value):
        self.put(value)

    def put(self, value, wait=False):
        self.write_attribute(self.propty, value)

    def wait_for_connection(self, timeout=None):
        pass
