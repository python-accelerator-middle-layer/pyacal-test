"""."""

from .. import _get_facility
from .base import Device


class BPM(Device):
    """."""

    PROPERTIES_DEFAULT = ("posx", "posy")

    def __init__(self, devname):
        """."""
        facil = _get_facility()
        if not facil.is_alias_in_cs_devtype(devname, facil.CSDevTypes.BPM):
            raise ValueError(f"Device name: {devname} not valid for a BPM.")

        super().__init__(devname, props2init=BPM.PROPERTIES_DEFAULT)

    @property
    def posx(self):
        """."""
        return self["posx"]

    @property
    def posy(self):
        """."""
        return self["posy"]
