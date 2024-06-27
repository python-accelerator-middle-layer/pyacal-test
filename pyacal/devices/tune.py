"""."""

from .. import _get_facility
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

        facil = _get_facility()
        if "Tune" not in facil.alias_map[devname]["cs_devtype"]:
            raise ValueError(f"Device name: {devname} not valid for Tune")

    @property
    def tunex(self):
        """."""
        return self["tunex"]

    @property
    def tuney(self):
        """."""
        return self["tuney"]
