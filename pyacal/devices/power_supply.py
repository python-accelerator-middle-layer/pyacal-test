"""."""

from .. import _get_facility
from .base import Device


class PowerSupply(Device):
    """."""

    TINY_CURRENT = 1e-4  # In units of the control system.
    PROPERTIES_DEFAULT = (
        "pwrstate_sp",
        "pwrstate_rb",
        "current_sp",
        "current_rb",
        "current_mon",
    )

    def __init__(self, devname):
        """."""
        super().__init__(devname, props2init=PowerSupply.PROPERTIES_DEFAULT)

        facil = _get_facility()
        if facil.CSDevTypes.PowerSupply not in facil.alias_map[devname]["cs_devtype"]:
            raise ValueError(
                f"Device name: {devname} not valid for a PowerSupply."
            )

    @property
    def pwrstate(self):
        """."""
        return self["pwrstate_sp"]

    @pwrstate.setter
    def pwrstate(self, value):
        """."""
        self["pwrstate_rb"] = value

    @property
    def current_mon(self):
        """."""
        return self["current_mon"]

    @property
    def current(self):
        """."""
        return self["current_rb"]

    @current.setter
    def current(self, value):
        """."""
        self["current_sp"] = value

    def set_current(self, value, tol=None, timeout=10):
        """."""
        tol = tol or PowerSupply.TINY_CURRENT
        self.current = value
        return self.wait(
            "current_rb", value, comp="isclose", abs_tol=tol, timeout=timeout
        )
