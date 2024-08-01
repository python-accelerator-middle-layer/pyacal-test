import time as _time

from .. import _get_control_system, _get_facility
from .converters import create_converter
from .utils import ConverterTypes


class PV:

    def __init__(self, devname, propty, connection_timeout=None):
        """."""
        self.devname = devname
        self.propty = propty
        control = _get_control_system()
        self._key = control.create_pv(
            self.devname,
            self.propty,
            connection_timeout=connection_timeout,
        )
        self._converters = self._create_converters()

    @property
    def connected(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        conn = pvo.connected
        for conv in self._converters:
            conn &= conv.connected
        return conn

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
        pvo = _get_control_system().get_pv(self._key)
        return pvo.timestamp

    @property
    def host(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.host

    @property
    def units(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.units

    @property
    def lower_limit(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return self._conversion_forward(pvo.lower_limit)

    @property
    def upper_limit(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return self._conversion_forward(pvo.upper_limit)

    def get(self, timeout=None):
        """."""
        if not self.wait_for_connection(timeout=timeout):
            return None
        pvo = _get_control_system().get_pv(self._key)
        return self._conversion_forward(pvo.get(timeout=timeout))

    def put(self, value, wait=False):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.put(self._conversion_reverse(value), wait=wait)

    def wait_for_connection(self, timeout=None):
        """."""
        t0_ = _time.time()
        pvo = _get_control_system().get_pv(self._key)
        conn = pvo.wait_for_connection(timeout=timeout)
        dt_ = _time.time() - t0_
        for conv in self._converters:
            time_left = timeout if timeout is None else timeout - dt_
            if timeout is not None and time_left <= 0:
                return False
            conn &= conv.wait_for_connection(timeout=time_left)
            dt_ = _time.time() - t0_
        return conn

    # ---------------------- helper methods -----------------------
    def _conversion_forward(self, value):
        """."""
        for conv in self._converters:
            value = conv.conversion_forward(value)
        return value

    def _conversion_reverse(self, value):
        """."""
        for conv in self._converters[::-1]:
            value = conv.conversion_reverse(value)
        return value

    def _create_converters(self):
        """."""
        fac = _get_facility()
        conv_list = fac.get_attribute_from_aliases(
            f'cs_propties.{self.propty}.conv_cs2phys',
            self.devname,
        )
        convs = []
        for c_def in conv_list:
            conv_type = c_def['type']
            if conv_type not in ConverterTypes:
                raise ValueError(f'Converter {conv_type} not defined.')
            props = ConverterTypes[conv_type]
            conv = create_converter(
                conv_type,
                kwargs={k: v for k, v in c_def.items() if k in props}
            )
            convs.append(conv)
        return convs
