from .. import ALIAS_MAP

from .base import Device
from .fambpms import FamBPMs
from .famcms import FamCMs


class SOFB(Device):
    """."""

    PROPERTIES_DEFAULT = (
        "orbx",
        "orby",
        "refx",
        "refy",
        "nr_points",
        "kickch",
        "kickcv",
    )

    def __init__(self, devname):
        """."""
        super().__init__(
            devname,
            props2init=SOFB.PROPERTIES_DEFAULT,
        )

        self.fambpms = FamBPMs()
        self.famcms = FamCMs()
        self.nbpms = len(self.fambpms.devices)
        self.nhcms = self.famcms.nhcms
        self.nvcms = self.famcms.nvcms

    @property
    def orbx(self):
        """."""
        _orb = self.fambpms.get_orbit()
        return _orb[: self.nbpms // 2]

    @property
    def orby(self):
        """."""
        _orb = self.fambpms.get_orbit()
        return _orb[self.nbpms // 2 :]

    @property
    def kickch(self):
        """."""
        _kicks = self.famcms.get_currents()
        return _kicks[: self.nhcms]

    @kickch.setter
    def kickch(self, values):
        for i, hcm in enumerate(self.famcms.hcms):
            hcm.current = values[i]

    @property
    def kickcv(self):
        """."""
        _kicks = self.famcms.get_currents()
        return _kicks[self.nhcms : self.nhcms + self.nvcms]

    @kickcv.setter
    def kickcv(self, values):
        for i, vcm in enumerate(self.famcms.vcms):
            vcm.current = values[i]
