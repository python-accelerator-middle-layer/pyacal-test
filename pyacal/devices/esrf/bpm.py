"""."""

from ... import _get_facility
from ..base import Device


class BPM(Device):
    """."""

    PROPERTIES_DEFAULT = ("posx", "posy")

    def __init__(self, devname, auto_monitor_mon=True):
        """."""
        facil = _get_facility()
        ds_info = facil.get_attribute_from_aliases('ds_info', devname)
        self.idx = ds_info['vector_index']
        if not facil.is_alias_in_cs_devtype(devname, facil.CSDevTypes.BPM):
            raise ValueError(f"Device name: {devname} not valid for a BPM.")
        super().__init__(devname, props2init=BPM.PROPERTIES_DEFAULT, )

    @property
    def posx(self):
        """."""
        val = self["posx"]
        return val if self.idx is None else val[self.idx]


    @property
    def posy(self):
        """."""
        val = self["posy"]
        return val if self.idx is None else val[self.idx]
