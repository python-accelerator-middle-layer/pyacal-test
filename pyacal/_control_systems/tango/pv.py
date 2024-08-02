"""PV class."""

import tango

from ... import _get_facility


class PV:
    """PV class."""

    def __init__(self, devname, propty, connection_timeout=None):
        """."""
        self.devname = devname
        self.propty = propty
        facil = _get_facility()
        self.cs_devname = facil.get_attribute_from_aliases(
            'cs_devname', devname
        )
        properties = facil.get_attribute_from_aliases(
            f'cs_propties.{propty}', devname
        )
        self.cs_propty = properties['name']
        self.wvalue = properties.get('wvalue', False)
        self.idx = properties.get('index', None)
        try:
            self._ds = facil._CONNECTED_DS[self.cs_devname]
        except KeyError:
            self._ds = tango.DeviceProxy(self.cs_devname)
            facil._CONNECTED_DS[self.cs_devname] = self._ds
        self.config = self._ds.get_attribute_config(self.cs_propty)
        self._timestamp = None
        self._pvname = self.cs_devname + '/' + self.cs_propty

    @property
    def pvname(self):
        return self._pvname

    @property
    def connected(self):
        """."""
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
        attr = self._ds.read_attribute(self.cs_propty)
        self._timestamp = attr.time.totime()
        value = attr.w_value if self.wvalue else attr.value
        return value if self.idx is None else value[self.idx]

    def put(self, value, wait=None):
        self._ds.write_attribute(self.cs_propty, value)

    def wait_for_connection(self, timeout=None):
        pass
