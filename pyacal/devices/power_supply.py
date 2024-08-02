"""."""

from .. import _get_facility
from .base import Device


class PowerSupply(Device):
    """."""

    TINY_STRENGTH = 1e-3
    PROPERTIES_DEFAULT = (
        "pwrstate_sp",
        "pwrstate_rb",
        "strength_sp",
        "strength_rb",
        "strength_mon",
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
        return self["pwrstate_rb"]

    @pwrstate.setter
    def pwrstate(self, value):
        """."""
        self["pwrstate_sp"] = value

    @property
    def strength_mon(self):
        """."""
        return self["strength_mon"]

    @property
    def strength(self):
        """."""
        return self["strength_rb"]

    @strength.setter
    def strength(self, value):
        """."""
        self["strength_sp"] = value

    def set_strength(self, value, tol=None, timeout=10):
        """."""
        tol = tol or self.TINY_STRENGTH
        self.strength = value
        return self.wait(
            "strength_rb", value, comp="isclose", abs_tol=tol, timeout=timeout
        )
