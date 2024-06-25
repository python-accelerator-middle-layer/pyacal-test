from .. import ALIAS_MAP

from .base import Device


class BPM(Device):
    """."""

    PROPERTIES_DEFAULT = ("posx", "posy")

    def __init__(self, devname, auto_monitor_mon=True):
        """."""
        super().__init__(
            devname,
            props2init=BPM.PROPERTIES_DEFAULT,
            auto_monitor_mon=auto_monitor_mon,
        )

        if "BPM" not in ALIAS_MAP[devname]["info"]:
            raise ValueError(f"Device name: {devname} not valid for a BPM.")

    @property
    def posx(self):
        """."""
        return self["posx"]

    @property
    def posy(self):
        """."""
        return self["posy"]
