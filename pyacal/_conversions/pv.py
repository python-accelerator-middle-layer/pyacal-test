from .. import _get_control_system


class PV:

    def __init__(
        self, devname, propty, auto_monitor=True, connection_timeout=None
    ):
        """."""
        self.devname = devname
        self.propty = propty
        control = _get_control_system()
        self._key = control.create_pv(
            self.devname,
            self.propty,
            auto_monitor=auto_monitor,
            connection_timeout=connection_timeout,
        )
        self._converters = self._create_converters()

    @property
    def connected(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.connected

    @property
    def auto_monitor(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.auto_monitor

    @auto_monitor.setter
    def auto_monitor(self, value):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        pvo.auto_monitor = int(value)

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
    def precision(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.precision

    @property
    def lower_limit(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return self._convert_direct(pvo.lower_limit)

    @property
    def upper_limit(self):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return self._convert_direct(pvo.upper_limit)

    def get(self, timeout=None):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return self._convert_direct(pvo.get(timeout=timeout))

    def put(self, value, wait=False):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.put(self._convert_reverse(value), wait=wait)

    def wait_for_connection(self, timeout=None):
        """."""
        pvo = _get_control_system().get_pv(self._key)
        return pvo.wait_for_connection(timeout=timeout)

    # ---------------------- helper methods -----------------------
    def _convert_direct(self, value):
        """."""
        return value

    def _convert_reverse(self, value):
        """."""
        return value

    def _create_converters(self):
        """."""
        return []
