"""."""

from .. import _get_facility
from .base import Device


class Tune(Device):
    """."""

    PROPERTIES_DEFAULT = ("tunex", "tuney")

    def __init__(self, devname=None, accelerator=None):
        """."""
        fac = _get_facility()
        self.accelerator = accelerator or fac.default_accelerator
        if devname is None:
            devname = fac.find_aliases_from_cs_devtype(fac.CSDevTypes.TuneMeas)
            devname = fac.find_aliases_from_accelerator(
                self.accelerator, devname)[0]

        if fac.is_alias_in_cs_devtype(devname, fac.CSDevTypes.TuneMeas):
            raise ValueError(f"Device name: {devname} not valid for Tune")

        super().__init__(
            devname, props2init=Tune.PROPERTIES_DEFAULT,
        )

    @property
    def tunex(self):
        """."""
        return self["tunex"]

    @property
    def tuney(self):
        """."""
        return self["tuney"]
