from .. import ALIAS_MAP

from .base import Device

MIN_CURRENT = 0.01  # [mA]


class DCCT(Device):
    """."""

    PROPERTIES_DEFAULT = ("current",)

    def __init__(self, devname):
        """."""
        super().__init__(devname, props2init=DCCT.PROPERTIES_DEFAULT)

        if "DCCT" not in ALIAS_MAP[devname]["info"]:
            raise ValueError(f"Device name: {devname} not valid for a DCCT.")

    @property
    def current(self):
        """."""
        return self["current"]

    @property
    def storedbeam(self):
        """."""
        return self.current > MIN_CURRENT
