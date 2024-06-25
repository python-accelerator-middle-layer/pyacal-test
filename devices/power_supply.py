from .. import FACILITY

from .base import Device


class PowerSupply(Device):
    """."""

    PROPERTIES_DEFAULT = (
        "pwrstate_sp",
        "pwrstate_rb",
        "current_sp",
        "current_rb",
    )

    def __init__(self, devname):
        """."""
        super().__init__(devname, props2init=PowerSupply.PROPERTIES_DEFAULT)

        if "PowerSupply" not in FACILITY.alias_map[devname]["cs_devtype"]:
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
    def current(self):
        """."""
        return self["current_rb"]

    @current.setter
    def current(self, value):
        """."""
        self["current_sp"] = value
