"""."""

from .. import FACILITY
from .base import Device


class Tune(Device):
    """."""

    PROPERTIES_DEFAULT = ("tunex", "tuney")

    def __init__(self, devname):
        """."""
        super().__init__(
            devname,
            props2init=Tune.PROPERTIES_DEFAULT,
        )

        if "Tune" not in FACILITY.alias_map[devname]["cs_devtype"]:
            raise ValueError(f"Device name: {devname} not valid for Tune")

    @property
    def tunex(self):
        """."""
        return self["tunex"]

    @property
    def tuney(self):
        """."""
        return self["tuney"]
