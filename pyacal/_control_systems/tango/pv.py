"""PV class."""

import tango
from ... import _get_facility


class PV:
    """PV class."""

    def __init__(self, devname, propty, auto_monitor=True, connection_timeout=None):
        """."""
        self.devname = devname
        self.propty = propty
        facil = _get_facility()
        self.cs_devname = facil.get_attribute_from_aliases('cs_devname', devname)
        self.isvector = False
        if 'all' in self.cs_devname:
            self.isvector = True
            self.vector_idx = facil.get_attribute_from_aliases('vector_index', devname)
        self.cs_propty = facil.get_attribute_from_aliases(
            f'cs_propties.{propty}.name', devname)
        try:
            self._ds = facil._CONNECTED_DS[devname]
        except KeyError:
            self._ds = tango.DeviceProxy(self.cs_devname)
            facil._CONNECTED_DS[devname] = self._ds
        self.config = self._ds.get_attribute_config(self.cs_propty)
        self._timestamp = None
        self._pvname = self.cs_devname + '/' + self.cs_propty

    @property
    def pvname(self):
        return self._pvname

    @property
    def connected(self):
        try:
            self._ds.state()
            return True
        except tango.DevFailed as e:
            print(e)
            return False

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
        return self._timestamp

    @property
    def host(self):
        """."""
        return self._ds.get_db_host()

    @property
    def units(self):
        """."""
        return self.config.unit

    @property
    def lower_limit(self):
        """."""
        return self.config.min_value

    @property
    def upper_limit(self):
        """."""
        return self.config.max_value

    def get(self, timeout=None):
        """."""
        attr = self._ds.read_attribute(self.cs_propty)
        self._timestamp = attr.time.totime()
        if '_mon' in self.propty:
            attrv = attr.value
        else:
            cond = (attr.value is None) or (attr.value == [])
            attrv = attr.w_value if cond else attr.value
        if self.isvector:
            return attrv[self.vector_idx]
        else:
            return attrv

    def put(self, value, wait=None):
        if self.isvector:
            v0 = self._ds.read_attribute(self.cs_propty).w_value
            v0[self.vector_idx] = value
            self._ds.write_attribute(self.cs_propty, v0)
        else:
            self._ds.write_attribute(self.cs_propty, value)

    def wait_for_connection(self, timeout=None):
        pass
