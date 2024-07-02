"""."""

from .. import _get_facility
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

        facil = _get_facility()
        if facil.check_alias_in_csdevtype(devname, facil.CSDevType.BPM):
            raise ValueError(f"Device name: {devname} not valid for a BPM.")

    @property
    def posx(self):
        """."""
        return self["posx"]

    @property
    def posy(self):
        """."""
        return self["posy"]
