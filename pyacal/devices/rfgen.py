"""."""

import time as _time

import numpy as _np

from .. import _get_facility
from .base import Device


class RFGen(Device):
    """."""

    # NOTE: The properties below are used by the setter of the frequency
    # property to  not allow large abrupt frequency variations:
    RF_DELTA_MIN = 0.01  # in units of the control system
    RF_DELTA_MAX = 15000.0  # in units of the control system
    RF_DELTA_RMP = 200  # in units of the control system

    PROPERTIES_DEFAULT = ("frequency_rb", "frequency_sp")

    def __init__(self, devname=None):
        """."""
        fac = _get_facility()
        if devname is None:
            devname = fac.find_aliases_from_cs_devtype(
                fac.CSDevTypes.RFGenerator)[0]

        if not fac.is_alias_in_cs_devtype(devname, fac.CSDevTypes.RFGenerator):
            raise ValueError(f"Device name: {devname} not valid for a RFGen")

        super().__init__(devname, props2init=RFGen.PROPERTIES_DEFAULT)

    @property
    def frequency(self):
        """."""
        return self["frequency_rb"]

    @frequency.setter
    def frequency(self, value):
        """."""
        delta_max = RFGen.RF_DELTA_RMP  # in units of the control system
        freq0 = self.frequency
        if freq0 is None or value is None:
            return
        delta = abs(value-freq0)
        if delta < RFGen.RF_DELTA_MIN or delta > RFGen.RF_DELTA_MAX:
            return
        npoints = int(delta/delta_max) + 2
        freq_span = _np.linspace(freq0, value, npoints)[1:]
        pvo = self.pv_object('frequency_sp')
        pvo.put(freq_span[0], wait=False)
        for freq in freq_span[1:]:
            _time.sleep(1.0)
            pvo.put(freq, wait=False)
        self['frequency_sp'] = value

    def set_frequency(self, value, tol=1, timeout=10):
        """Set RF frequency and wait until it gets there."""
        self.frequency = value
        return self.wait(
            "frequency_rb", value, comp='isclose', abs_tol=tol, timeout=timeout
        )
