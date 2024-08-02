"""."""

from ... import _get_facility
from ..base import Device
import tango


class PowerSupply(Device):
    """."""

    TINY_CURRENT = 1e-6  # In units of the control system.
    PROPERTIES_DEFAULT = (
        "state",
        "strength_read",
        "strength_write"
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
    def strength_mon(self):
        """."""
        return self["strength_read"]

    @property
    def strength(self):
        """."""
        return self["strength_write"]

    @strength.setter
    def strength(self, value):
        """."""
        self["strength_write"] = value

    def set_strength(self, value, tol=None, timeout=10):
        """."""
        tol = tol or self.TINY_STRENGTH
        self.strength = value
        return self.wait(
            "strength_rb",
            value,
            comp="isclose",
            abs_tol=tol,
            timeout=timeout,
        )
