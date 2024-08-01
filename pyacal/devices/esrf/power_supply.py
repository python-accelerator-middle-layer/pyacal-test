"""."""

from ... import _get_facility
from ..base import Device
import tango


class PowerSupply(Device):
    """."""

    TINY_CURRENT = 1e-6  # In units of the control system.
    PROPERTIES_DEFAULT = (
        "state",
        "strength",
    )

    def __init__(self, devname):
        """."""
        fac = _get_facility()
        if not fac.is_alias_in_cs_devtype(devname, fac.CSDevTypes.PowerSupply):
            raise ValueError(
                f"Device name: {devname} not valid for a PowerSupply."
            )
        super().__init__(devname, props2init=PowerSupply.PROPERTIES_DEFAULT)

    @property
    def pwrstate(self):
        """."""
        state = self["state"]
        return True if state==tango.DevState.ON else False

    @property
    def current_mon(self):
        """."""
        val, _ = self["strength"]
        return val

    @property
    def current(self):
        """."""
        _, val = self["strength"]
        return val

    @current.setter
    def current(self, value):
        """."""
        self["strength"] = value

    def set_current(self, value, tol=None, timeout=10):
        """."""
        tol = tol or self.TINY_CURRENT
        self.current = value
        return self.wait(
            "current", value, comp="isclose", abs_tol=tol, timeout=timeout
        )
