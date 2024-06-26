"""."""

from .. import FACILITY
from .base import Device


class RFGen(Device):
    """."""

    PROPERTIES_DEFAULT = ("frequency-RB", "frequency-SP")

    def __init__(self, devname):
        """."""
        super().__init__(devname, props2init=RFGen.PROPERTIES_DEFAULT)

        if "RfGen" not in FACILITY.alias_map[devname]["cs_devtype"]:
            raise ValueError(f"Device name: {devname} not valid for a RFGen")

    @property
    def frequency(self):
        """."""
        return self["frequency-RB"]

    @frequency.setter
    def frequency(self, value):
        """."""
        self["frequency-SP"] = value

    def set_frequency(self, value, tol=1, timeout=10):
        """Set RF frequency and wait until it gets there."""
        self.frequency = value
        return self.wait_float(
            "frequency-RB", value, abs_tol=tol, timeout=timeout
        )
