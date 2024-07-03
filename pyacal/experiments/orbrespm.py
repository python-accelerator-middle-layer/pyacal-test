"""."""

import datetime as _datetime
import time as _time

import numpy as _np

from .. import _get_facility, get_alias_from_devtype as _get_alias_from_devtype
from ..devices import DCCT as _DCCT, SOFB as _SOFB
from .base import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _BaseClass,
)


class OrbRespmParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.delta_freq = 100  # [Hz]
        self.delta_curr_hcm = 1  # [A]
        self.delta_curr_vcm = 1  # [A]
        self.sofb_nrpoints = 10

    def __str__(self):
        """."""
        ftmp = "{0:24s} = {1:9.3f}  {2:s}\n".format
        dtmp = "{0:24s} = {1:9d}  {2:s}\n".format
        stg = ftmp("delta_freq", self.delta_freq, "[Hz]")
        stg += ftmp("delta_curr_hcm", self.delta_curr_hcm, "[A]")
        stg += ftmp("delta_curr_vcm", self.delta_curr_vcm, "[A]")
        stg += dtmp("sofb_nrpoints", self.sofb_nrpoints, "")
        return stg


class OrbRespm(_BaseClass):
    """."""

    def __init__(self, accelerator=None, isonline=True):
        """."""
        super().__init__(
            params=OrbRespmParams(), target=self._meas_respm, isonline=isonline
        )

        self.accelerator = accelerator or _get_facility().default_accelerator
        if self.isonline:
            self.devices["sofb"] = _SOFB(self.accelerator)
            dcct_alias = _get_alias_from_devtype("DCCT", self.accelerator)[0]
            self.devices["dcct"] = _DCCT(dcct_alias)

    def _meas_respm(self):
        tini = _datetime.datetime.fromtimestamp(_time.time())
        print(
            "Starting measurement at {:s}".format(
                tini.strftime("%Y-%m-%d %Hh%Mm%Ss")
            )
        )

        sofb = self.devices["sofb"]
        sofb.orb_nrpoints = self.params.sofb_nrpoints

        cm_enb = _np.r_[sofb.hcm_enbl, sofb.vcm_enbl]
        respm = _np.zeros((2 * sofb.nr_bpms, sofb.nr_cors), dtype=float)
        nr_enbl = len(cm_enb)

        # corrector magnets
        for i, enbl in enumerate(cm_enb):
            if not self._ok_to_continue():
                break

            print("\n{0:03d}/{1:03d}".format(i + 1, nr_enbl))
            if not enbl:
                print("     CM not enabled at SOFB. Skipping.")
                continue

            delta = (
                self.params.delta_curr_hcm
                if i < sofb.nr_hcms
                else self.params.delta_curr_vcm
            )
            cmdev = sofb.famcms.devices[i]
            success, respm_col = self._meas_respm_single_cm(cmdev, delta)
            if success:
                respm[:, i] = respm_col

        # rf frequency
        if not sofb.rfg_enbl:
            print("     RF not enabled at SOFB. Skipping.")
        else:
            if self._ok_to_continue():
                print("\n{0:03d}/{1:03d}".format(i + 1, nr_enbl))
                respm[:, -1] = self._meas_respm_rf()

        self.data["timestamp"] = _time.time()
        self.data["orbrespm"] = respm
        self.data["hcm_enbl"] = sofb.hcm_enbl
        self.data["vcm_enbl"] = sofb.vcm_enbl
        self.data["rfg_enbl"] = sofb.rfg_enbl

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split(".")[0]
        print("finished! Elapsed time {:s}".format(dtime))

    def _meas_respm_single_cm(self, cmdev, delta):
        sofb = self.devices["sofb"]
        cmname = cmdev.devname

        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime("%Hh%Mm%Ss")
        print("{:s} --> Measuring OrbRespm for {:s}".format(strtini, cmname))

        if not cmdev.pwrstate:
            print("\n    error: CM " + cmname + " is Off.")
            self._stopevt.set()
            print("    exiting...")
            return False, None

        steps = [-1, +1]
        total_step = steps[1] - steps[0]
        orbs = []
        inicurr = cmdev.current
        for step in steps:
            cmdev.set_current(inicurr + step * delta)
            orbs.append(sofb.get_orbit())
        cmdev.set_current(inicurr)

        return True, (orbs[1] - orbs[0]) / (total_step * delta)

    def _meas_respm_rf(self):
        sofb = self.devices["sofb"]
        rfgen = sofb.rfgen

        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime("%Hh%Mm%Ss")
        print("{:s} --> Measuring OrbRespm for: {:s}".format(strtini, "RF"))

        steps = [-1, +1]
        total_step = steps[1] - steps[0]
        orbs = []
        inifreq = rfgen.frequency
        delta = self.params.delta_freq
        for step in steps:
            rfgen.set_frequency(inifreq + step * delta)
            orbs.append(sofb.get_orbit())

        rfgen.set_frequency(inifreq)
        return (orbs[1] - orbs[0]) / (total_step * delta)

    def _ok_to_continue(self):
        if self._stopevt.is_set():
            print("stopped!")
            return False
        if not self.devices["dcct"].havebeam:
            print("Beam lost!")
            return False
        return True
