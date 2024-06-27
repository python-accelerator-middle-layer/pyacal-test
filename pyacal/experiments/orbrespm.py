"""."""

import datetime as _datetime
import time as _time

from .. import _get_facility
from .. import get_alias_from_devtype as _get_alias_from_devtype
from ..devices import DCCT as _DCCT, SOFB as _SOFB
from .base import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _BaseClass,
)

import numpy as _np


class OrbRespmParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.delta_freq = 100  # [Hz]
        self.delta_curr_hcm = 1  # [A]
        self.delta_curr_vcm = 1  # [A]
        self.sofb_nrpoints = 10
        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]

    def __str__(self):
        """."""
        ftmp = "{0:24s} = {1:9.3f}  {2:s}\n".format
        dtmp = "{0:24s} = {1:9d}  {2:s}\n".format
        stg = ftmp("delta_freq [Hz]", self.delta_freq, "")
        stg = ftmp("delta_curr_hcm [A]", self.delta_curr_hcm, "")
        stg = ftmp("delta_curr_vcm [A]", self.delta_curr_vcm, "")
        stg += dtmp("sofb_nrpoints", self.sofb_nrpoints, "")
        stg += dtmp("sofb_maxcorriter", self.sofb_maxcorriter, "")
        stg += ftmp("sofb_maxorberr [um]", self.sofb_maxorberr, "")
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

    @property
    def havebeam(self):
        """."""
        haveb = self.devices["dcct"]
        return haveb.connected and haveb.storedbeam

    def _meas_respm(self):
        tini = _datetime.datetime.fromtimestamp(_time.time())
        print(
            "Starting measurement at {:s}".format(
                tini.strftime("%Y-%m-%d %Hh%Mm%Ss")
            )
        )

        sofb = self.devices["sofb"]
        sofb.orb_nrpoints = self.params.sofb_nrpoints
        sofb.corr_gain_hcm = 100
        sofb.corr_gain_vcm = 100
        sofb.corr_gain_rfg = 100
        sofb.correct_orbit(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr,
        )

        cm_enb = _np.r_[sofb.hcm_enbl, sofb.vcm_enbl]
        respm = _np.zeros((2 * sofb.nr_bpms, sofb.nr_cors), dtype=float)
        nr_enbl = len(cm_enb)

        # corrector magnets
        for i, enbl in enumerate(cm_enb):
            if not self._ok_to_continue():
                break
            print("\n{0:03d}/{1:03d}".format(i + 1, nr_enbl))
            if enbl:
                cmname = sofb.famcms.cm_names[i]
                cmtype = "hcm" if "h" in cmname.lower() else "vcm"
                success, respm_col = self._meas_respm_single_cm(cmname, cmtype)
                if success:
                    respm[:, i] = respm_col
            else:
                print("     CM not enabled at SOFB. Skipping.")

        # rf frequency
        if sofb.rfg_enbl:
            if self._ok_to_continue():
                print("\n{0:03d}/{1:03d}".format(nr_enbl, nr_enbl))
                respm[:, -1] = self._meas_respm_rf()
        else:
            print("     RF not enabled at SOFB. Skipping.")

        self.data["timestamp"] = _time.time()
        self.data["orbrespm"] = respm
        self.data["hcm_enbl"] = sofb.hcm_enbl
        self.data["vcm_enbl"] = sofb.vcm_enbl
        self.data["rfg_enbl"] = sofb.rfg_enbl

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split(".")[0]
        print("finished! Elapsed time {:s}".format(dtime))

    def _meas_respm_single_cm(self, cmname, cmtype):
        sofb = self.devices["sofb"]
        cmidx = sofb.famcms.cm_names.index(cmname)

        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime("%Hh%Mm%Ss")
        print(
            "{:s} --> Measuring OrbRespm for {:03d}: {:s}".format(
                strtini, cmidx, cmname
            )
        )

        cmdev = sofb.famcms.devices[cmidx]
        if not cmdev.pwrstate:
            print("\n    error: CM " + cmname + " is Off.")
            self._stopevt.set()
            print("    exiting...")
            return False, None

        steps = [+1, -1, 0]
        orbs = []
        # avoid applying variation to RF frequency
        sofb.delta_frequency_rfg = 0

        dcurr = getattr(self.params, f"delta_curr_{cmtype}")
        deltas = _np.zeros(getattr(sofb, f"nr_{cmtype}"), dtype=float)
        if cmtype == "vcm":
            cmidx -= sofb.nr_hcms
        for step in steps:
            deltas[cmidx] = step * dcurr
            setattr(sofb, f"delta_currents_{cmtype}", deltas)
            sofb.apply_correction()
            orbs.append(sofb.get_orbit())
        return True, (orbs[0] - orbs[1]) / (2 * dcurr)

    def _meas_respm_rf(self):
        sofb = self.devices["sofb"]

        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime("%Hh%Mm%Ss")
        print(
            "{:s} --> Measuring OrbRespm for {:03d}: {:s}".format(
                strtini, len(sofb.nr_cors), "RF frequency"
            )
        )

        steps = [+1, -1, 0]
        orbs = []
        # avoid applying variations to CMs
        sofb.delta_currents_hcm *= 0
        sofb.delta_currents_vcm *= 0
        dfreq = self.params.delta_freq
        for step in steps:
            sofb.delta_frequency_rfg = step * dfreq
            sofb.apply_correction()
            orbs.append(sofb.get_orbit())
        return (orbs[0] - orbs[1]) / (2 * dfreq)

    def _ok_to_continue(self):
        if self._stopevt.is_set():
            print("stopped!")
            return False
        if not self.havebeam:
            print("Beam lost!")
            return False
        return True
