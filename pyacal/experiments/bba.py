"""Main module."""

import datetime as _datetime
import time as _time
from copy import deepcopy as _dcopy

import matplotlib.cm as _cmap
import matplotlib.gridspec as _mpl_gs
import matplotlib.pyplot as _plt
import numpy as _np
from scipy.optimize import least_squares as _least_squares

from .. import _get_facility, _get_simulator, \
    get_alias_from_devtype as _get_alias_from_devtype, \
    get_alias_from_indices as _get_alias_from_indices, \
    get_alias_map as _get_alias_map, \
    get_indices_from_key as _get_indices_from_key
from ..devices import DCCT as _DCCT, PowerSupply as _PowerSupply, SOFB as _SOFB
from .base import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _BaseClass


class BBAParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.deltaorbx = 100  # [um]
        self.deltaorby = 100  # [um]
        self.meas_nrsteps = 8
        self.quad_deltacurr = 1  # [A]
        self.quad_maxcurr = 200  # [A]
        self.quad_mincurr = 0  # [A]
        self.quad_nrcycles = 1
        self.wait_correctors = 0.3  # [s]
        self.wait_quadrupole = 0.3  # [s]
        self.timeout_wait_orbit = 3  # [s]
        self.sofb_nrpoints = 10
        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stg = ftmp('deltaorbx [um]', self.deltaorbx, '')
        stg += ftmp('deltaorby [um]', self.deltaorby, '')
        stg += dtmp('meas_nrsteps', self.meas_nrsteps, '')
        stg += ftmp('quad_deltacurr [A]', self.quad_deltacurr, '')
        stg += ftmp('quad_maxcurr [A]', self.quad_maxcurr, '')
        stg += ftmp('quad_mincurr [A]', self.quad_mincurr, '')
        stg += ftmp('quad_nrcycles', self.quad_nrcycles, '')
        stg += ftmp('wait_correctors [s]', self.wait_correctors, '')
        stg += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        stg += ftmp(
            'timeout_wait_orbit [s]', self.timeout_wait_orbit, '(get orbit)')
        stg += dtmp('sofb_nrpoints', self.sofb_nrpoints, '')
        stg += dtmp('sofb_maxcorriter', self.sofb_maxcorriter, '')
        stg += ftmp('sofb_maxorberr [um]', self.sofb_maxorberr, '')
        return stg


class BBA(_BaseClass):
    """."""

    def __init__(self, accelerator=None, isonline=True):
        """."""
        super().__init__(
            params=BBAParams(), target=self._meas_bba, isonline=isonline
        )
        self._bpms2dobba = list()
        self.data["measure"] = dict()
        self._amap = _get_alias_map()

        self.accelerator = accelerator or _get_facility().default_accelerator
        if self.isonline:
            self.devices["sofb"] = _SOFB(self.accelerator)
            dcct_alias = _get_alias_from_devtype("DCCT", self.accelerator)[0]
            self.devices["dcct"] = _DCCT(dcct_alias)

            self.data["bpmnames"] = self.devices["sofb"].fambpms.bpm_names
            self.data["quadnames"] = self.get_default_quads()
            self.connect_to_quadrupoles()

        if "bpmnames" in self.data:
            self.data["scancenterx"] = _np.zeros(len(self.data["bpmnames"]))
            self.data["scancentery"] = _np.zeros(len(self.data["bpmnames"]))

    def __str__(self):
        """."""
        stn = 'Params\n'
        stp = self.params.__str__()
        stp = '    ' + stp.replace('\n', '\n    ')
        stn += stp + '\n'
        stn += 'Connected?  ' + str(self.connected) + '\n\n'

        stn += '     {:^20s} {:^20s} {:6s} {:6s}\n'.format(
            'BPM', 'Quad', 'Xc [um]', 'Yc [um]')
        tmplt = '{:03d}: {:^20s} {:^20s} {:^6.1f} {:^6.1f}\n'
        dta = self.data
        for bpm in self.bpms2dobba:
            idx = dta['bpmnames'].index(bpm)
            stn += tmplt.format(
                idx, dta['bpmnames'][idx], dta['quadnames'][idx],
                dta['scancenterx'][idx], dta['scancentery'][idx])
        return stn

    @property
    def measuredbpms(self):
        """."""
        return sorted(self.data['measure'])

    @property
    def bpms2dobba(self):
        """."""
        if self._bpms2dobba:
            return _dcopy(self._bpms2dobba)
        return sorted(
            set(self.data['bpmnames']) - self.data['measure'].keys())

    @bpms2dobba.setter
    def bpms2dobba(self, bpmlist):
        """."""
        self._bpms2dobba = [bpm for bpm in bpmlist]

    def connect_to_quadrupoles(self):
        """."""
        for bpm in self.bpms2dobba:
            idx = self.data['bpmnames'].index(bpm)
            qname = self.data['quadnames'][idx]
            if qname and qname not in self.devices:
                self.devices[qname] = _PowerSupply(qname)

    @staticmethod
    def get_cycling_curve():
        """."""
        return [1/2, -1/2, 0]

    def correct_orbit_at_bpm(self, bpmname, x0, y0):
        """."""
        sofb = self.devices['sofb']
        idxx = self.data['bpmnames'].index(bpmname)
        refx, refy = sofb.ref_orbx, sofb.ref_orby
        refx[idxx], refy[idxx] = x0, y0
        sofb.ref_orbx, sofb.ref_orby = refx, refy
        idx, resx, resy = sofb.correct_orbit(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr)
        return idx, _np.max([resx, resy])

    def correct_orbit(self):
        """."""
        self.devices['sofb'].correct_orbit(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr)

    def process_data(
            self, nbpms_linfit=None, thres=None, mode='symm',
            discardpoints=None, nonlinear=False):
        """."""
        for bpm in self.data['measure']:
            self.analysis[bpm] = self.process_data_single_bpm(
                bpm, nbpms_linfit=nbpms_linfit, thres=thres, mode=mode,
                discardpoints=discardpoints, nonlinear=nonlinear)

    def process_data_single_bpm(
            self, bpm, nbpms_linfit=None, thres=None, mode='symm',
            discardpoints=None, nonlinear=False):
        """."""
        anl = dict()
        idx = self.data['bpmnames'].index(bpm)
        nbpms = len(self.data['bpmnames'])
        orbini = self.data['measure'][bpm]['orbini']
        orbpos = self.data['measure'][bpm]['orbpos']
        orbneg = self.data['measure'][bpm]['orbneg']

        usepts = set(range(orbini.shape[0]))
        if discardpoints is not None:
            usepts = set(usepts) - set(discardpoints)
        usepts = sorted(usepts)

        xpos = orbini[usepts, idx]
        ypos = orbini[usepts, idx+nbpms]
        if mode.lower().startswith('symm'):
            dorb = orbpos - orbneg
        elif mode.lower().startswith('pos'):
            dorb = orbpos - orbini
        else:
            dorb = orbini - orbneg

        dorbx = dorb[usepts, :nbpms]
        dorby = dorb[usepts, nbpms:]
        devtype = self._amap[self.data["quadnames"][idx]]["cs_devtype"]
        if "skew" in devtype.lower():
            dorbx, dorby = dorby, dorbx
        anl['xpos'] = xpos
        anl['ypos'] = ypos

        px = _np.polyfit(xpos, dorbx, deg=1)
        py = _np.polyfit(ypos, dorby, deg=1)

        nbpms_linfit = nbpms_linfit or len(self.data['bpmnames'])
        sidx = _np.argsort(_np.abs(px[0]))
        sidy = _np.argsort(_np.abs(py[0]))
        sidx = sidx[-nbpms_linfit:][::-1]
        sidy = sidy[-nbpms_linfit:][::-1]
        pxc = px[:, sidx]
        pyc = py[:, sidy]
        if thres:
            ax2 = pxc[0]*pxc[0]
            ay2 = pyc[0]*pyc[0]
            ax2 /= ax2[0]
            ay2 /= ay2[0]
            nx = _np.sum(ax2 > thres)
            ny = _np.sum(ay2 > thres)
            pxc = pxc[:, :nx]
            pyc = pyc[:, :ny]

        x0s = -pxc[1]/pxc[0]
        y0s = -pyc[1]/pyc[0]
        x0 = _np.dot(pxc[0], -pxc[1]) / _np.dot(pxc[0], pxc[0])
        y0 = _np.dot(pyc[0], -pyc[1]) / _np.dot(pyc[0], pyc[0])
        stdx0 = _np.sqrt(
            _np.dot(pxc[1], pxc[1]) / _np.dot(pxc[0], pxc[0]) - x0*x0)
        stdy0 = _np.sqrt(
            _np.dot(pyc[1], pyc[1]) / _np.dot(pyc[0], pyc[0]) - y0*y0)
        extrapx = not min(xpos) <= x0 <= max(xpos)
        extrapy = not min(ypos) <= y0 <= max(ypos)
        anl['linear_fitting'] = dict()
        anl['linear_fitting']['dorbx'] = dorbx
        anl['linear_fitting']['dorby'] = dorby
        anl['linear_fitting']['coeffsx'] = px
        anl['linear_fitting']['coeffsy'] = py
        anl['linear_fitting']['x0s'] = x0s
        anl['linear_fitting']['y0s'] = y0s
        anl['linear_fitting']['extrapolatedx'] = extrapx
        anl['linear_fitting']['extrapolatedy'] = extrapy
        anl['linear_fitting']['x0'] = x0
        anl['linear_fitting']['y0'] = y0
        anl['linear_fitting']['stdx0'] = stdx0
        anl['linear_fitting']['stdy0'] = stdy0

        rmsx = _np.sum(dorbx*dorbx, axis=1) / dorbx.shape[1]
        rmsy = _np.sum(dorby*dorby, axis=1) / dorby.shape[1]
        if xpos.size > 3:
            px, covx = _np.polyfit(xpos, rmsx, deg=2, cov=True)
            py, covy = _np.polyfit(ypos, rmsy, deg=2, cov=True)
        else:
            px = _np.polyfit(xpos, rmsx, deg=2, cov=False)
            py = _np.polyfit(ypos, rmsy, deg=2, cov=False)
            covx = covy = _np.zeros((3, 3))

        x0 = -px[1] / px[0] / 2
        y0 = -py[1] / py[0] / 2
        stdx0 = _np.abs(x0)*_np.sqrt(_np.sum(_np.diag(covx)[:2]/px[:2]/px[:2]))
        stdy0 = _np.abs(y0)*_np.sqrt(_np.sum(_np.diag(covy)[:2]/py[:2]/py[:2]))

        if nonlinear:
            fitx = _least_squares(
                fun=lambda par, x, y: (y - par[1]*(x - par[0])**2),
                x0=[x0, px[0]], args=(xpos, rmsx), method='lm')
            fity = _least_squares(
                fun=lambda par, x, y: (y - par[1]*(x - par[0])**2),
                x0=[y0, py[0]], args=(ypos, rmsy), method='lm')
            x0, conx = fitx['x']
            y0, cony = fity['x']
            px = _np.array([conx, -2*conx*x0, conx*x0*x0])
            py = _np.array([cony, -2*cony*y0, cony*y0*y0])
            stdx0 = self._calc_fitting_error(fitx)[0]
            stdy0 = self._calc_fitting_error(fity)[0]

        extrapx = not min(xpos) <= x0 <= max(xpos)
        extrapy = not min(ypos) <= y0 <= max(ypos)
        anl['quadratic_fitting'] = dict()
        anl['quadratic_fitting']['meansqrx'] = rmsx
        anl['quadratic_fitting']['meansqry'] = rmsy
        anl['quadratic_fitting']['coeffsx'] = px
        anl['quadratic_fitting']['coeffsy'] = py
        anl['quadratic_fitting']['extrapolatedx'] = extrapx
        anl['quadratic_fitting']['extrapolatedy'] = extrapy
        anl['quadratic_fitting']['x0'] = x0
        anl['quadratic_fitting']['y0'] = y0
        anl['quadratic_fitting']['stdx0'] = stdx0
        anl['quadratic_fitting']['stdy0'] = stdy0
        return anl

    def get_bba_results(self, method='linear_fitting', error=False):
        """."""
        data = self.data
        bpms = data['bpmnames']
        bbax = _np.zeros(len(bpms))
        bbay = _np.zeros(len(bpms))
        if error:
            bbaxerr = _np.zeros(len(bpms))
            bbayerr = _np.zeros(len(bpms))
        for idx, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                continue
            res = anl[method]
            bbax[idx] = res['x0']
            bbay[idx] = res['y0']
            if error and 'stdx0' in res:
                bbaxerr[idx] = res['stdx0']
                bbayerr[idx] = res['stdy0']
        if error:
            return bbax, bbay, bbaxerr, bbayerr
        return bbax, bbay

    def get_analysis_properties(self, propty, method='linear_fitting'):
        """."""
        data = self.data
        bpms = data['bpmnames']
        prop = [[], ] * len(bpms)
        for idx, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                continue
            res = anl[method]
            prop[idx] = res[propty]
        return prop

    def get_default_quads(self):
        """."""
        bpmnames = self.data["bpmnames"]
        quads_idx = _get_indices_from_key("cs_devtype", "QuadrupoleNormal")
        qs_idx = _get_indices_from_key("cs_devtype", "QuadrupoleSkew")
        bpms_idx = []
        for bpmname in bpmnames:
            bpms_idx.append(
                [idx for idx in self._amap[bpmname]["sim_info"]["indices"]]
            )

        quads_idx.extend(qs_idx)
        quads_idx = _np.array([idx[len(idx) // 2] for idx in quads_idx])

        simul = _get_simulator()
        quads_pos = _np.array(simul.get_positions(
            acc=self.accelerator, indices=quads_idx)).ravel()
        bpms_pos = _np.array(simul.get_positions(
            acc=self.accelerator, indices=bpms_idx)).ravel()

        diff = _np.abs(bpms_pos[:, None] - quads_pos[None, :])
        bba_idx = _np.argmin(diff, axis=1)
        quads_bba_idx = quads_idx[bba_idx]
        qnames = list()
        for qidx in quads_bba_idx:
            qnames.append(_get_alias_from_indices(qidx)[0])
        return qnames

    @staticmethod
    def _calc_fitting_error(fit_params):
        # based on fitting error calculation of scipy.optimization.curve_fit
        # do Moore-Penrose inverse discarding zero singular values.
        _, smat, vhmat = _np.linalg.svd(
            fit_params['jac'], full_matrices=False)
        thre = _np.finfo(float).eps * max(fit_params['jac'].shape)
        thre *= smat[0]
        smat = smat[smat > thre]
        vhmat = vhmat[:smat.size]
        pcov = _np.dot(vhmat.T / (smat*smat), vhmat)

        # multiply covariance matrix by residue 2-norm
        ysize = len(fit_params['fun'])
        cost = 2 * fit_params['cost']  # res.cost is half sum of squares!
        popt = fit_params['x']
        if ysize > popt.size:
            # normalized by degrees of freedom
            s_sq = cost / (ysize - popt.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(0.0)
            print(
                '# of fitting parameters larger than # of data points!')
        return _np.sqrt(_np.diag(pcov))

    @staticmethod
    def _calc_dorb_scan(deltaorb, nrpts):
        dorbspos = _np.linspace(deltaorb, 0, nrpts+1)[:-1]
        dorbsneg = _np.linspace(-deltaorb, 0, nrpts+1)[:-1]
        dorbs = _np.array([dorbsneg, dorbspos]).T.ravel()
        dorbs = _np.hstack([0, dorbs])
        return dorbs

    @staticmethod
    def list_bpm_subsections(bpms):
        """."""
        subinst = [bpm[5:7]+bpm[14:] for bpm in bpms]
        sec = [bpm[2:4] for bpm in bpms]
        subsecs = {typ: [] for typ in subinst}
        for sub, sec, bpm in zip(subinst, sec, bpms):
            subsecs[sub].append(bpm)
        return subsecs

    def combine_bbas(self, bbalist):
        """."""
        items = ['quadnames', 'scancenterx', 'scancentery']
        dobba = BBA()
        dobba.params = self.params
        dobba.data = _dcopy(self.data)
        for bba in bbalist:
            for bpm, data in bba.data['measure'].items():
                dobba.data['measure'][bpm] = _dcopy(data)
                idx = dobba.data['bpmnames'].index(bpm)
                for item in items:
                    dobba.data[item][idx] = bba.data[item][idx]
        return dobba

    def filter_problems(
            self, maxstd=100, maxorb=9, maxrms=100, method='lin quad',
            probtype='std', pln='xy'):
        """."""
        bpms = []
        islin = 'lin' in method
        isquad = 'quad' in method
        for bpm in self.data['bpmnames']:
            anl = self.analysis.get(bpm)
            if not anl:
                continue
            concx = anl['quadratic_fitting']['coeffsx'][0]
            concy = anl['quadratic_fitting']['coeffsy'][0]
            probc = False
            if 'x' in pln:
                probc |= concx < 0
            if 'y' in pln:
                probc |= concy < 0

            rmsx = anl['quadratic_fitting']['meansqrx']
            rmsy = anl['quadratic_fitting']['meansqry']
            probmaxrms = False
            if 'x' in pln:
                probmaxrms |= _np.max(rmsx) < maxrms
            if 'y' in pln:
                probmaxrms |= _np.max(rmsy) < maxrms

            extqx = isquad and anl['quadratic_fitting']['extrapolatedx']
            extqy = isquad and anl['quadratic_fitting']['extrapolatedy']
            extlx = islin and anl['linear_fitting']['extrapolatedx']
            extly = islin and anl['linear_fitting']['extrapolatedy']
            probe = False
            if 'x' in pln:
                probe |= extqx or extlx
            if 'y' in pln:
                probe |= extqy or extly

            stdqx = isquad and anl['quadratic_fitting']['stdx0'] > maxstd
            stdqy = isquad and anl['quadratic_fitting']['stdy0'] > maxstd
            stdlx = islin and anl['linear_fitting']['stdx0'] > maxstd
            stdly = islin and anl['linear_fitting']['stdy0'] > maxstd
            probs = False
            if 'x' in pln:
                probs |= stdqx or stdlx
            if 'y' in pln:
                probs |= stdqy or stdly

            dorbx = anl['linear_fitting']['dorbx']
            dorby = anl['linear_fitting']['dorby']
            probmaxorb = False
            if 'x' in pln:
                probmaxorb |= _np.max(_np.abs(dorbx)) < maxorb
            if 'y' in pln:
                probmaxorb |= _np.max(_np.abs(dorby)) < maxorb

            prob = False
            if 'std' in probtype:
                prob |= probs
            if 'ext' in probtype:
                prob |= probe
            if 'conc' in probtype:
                prob |= probc
            if 'rms' in probtype:
                prob |= probmaxrms
            if 'orb' in probtype:
                prob |= probmaxorb
            if 'all' in probtype:
                prob = probs and probe and probc and probmaxrms and probmaxorb
            if 'any' in probtype:
                prob = probs or probe or probc or probmaxrms or probmaxorb
            if prob:
                bpms.append(bpm)
        return bpms

    # ##### Make Figures #####
    def make_figure_bpm_summary(self, bpm, save=False):
        """."""
        f = _plt.figure(figsize=(9.5, 9))
        gs = _mpl_gs.GridSpec(3, 2)
        gs.update(
            left=0.11, right=0.98, bottom=0.1, top=0.9,
            hspace=0.35, wspace=0.35)

        f.suptitle(bpm, fontsize=20)

        alx = _plt.subplot(gs[0, 0])
        aly = _plt.subplot(gs[0, 1])
        aqx = _plt.subplot(gs[1, 0])
        aqy = _plt.subplot(gs[1, 1])
        adt = _plt.subplot(gs[2, 0])
        axy = _plt.subplot(gs[2, 1])

        allax = [alx, aly, aqx, aqy, axy]

        for ax in allax:
            ax.grid(True)

        anl = self.analysis.get(bpm)
        if not anl:
            print('no dada found for ' + bpm)
            return
        xpos = anl['xpos']
        ypos = anl['ypos']
        sxpos = _np.sort(xpos)
        sypos = _np.sort(ypos)

        xq0 = anl['quadratic_fitting']['x0']
        yq0 = anl['quadratic_fitting']['y0']
        stdxq0 = anl['quadratic_fitting']['stdx0']
        stdyq0 = anl['quadratic_fitting']['stdy0']
        xl0 = anl['linear_fitting']['x0']
        yl0 = anl['linear_fitting']['y0']
        stdxl0 = anl['linear_fitting']['stdx0']
        stdyl0 = anl['linear_fitting']['stdy0']

        adt.set_frame_on(False)
        adt.axes.get_yaxis().set_visible(False)
        adt.axes.get_xaxis().set_visible(False)
        idx = self.data['bpmnames'].index(bpm)
        xini = self.data['scancenterx'][idx]
        yini = self.data['scancentery'][idx]
        qname = self.data['quadnames'][idx]

        currpos = self.data['measure'][bpm].get('currpos')
        currneg = self.data['measure'][bpm].get('currneg')
        if currpos is not None and currneg is not None:
            deltacurr = currpos - currneg
        else:
            deltacurr = self.data['measure'][bpm]['deltacurr']

        tmp = '{:6.1f} ' + r'$\pm$' + ' {:<6.1f}'
        st = 'Quad: {:15s} (dcurr={:.4f} A)\n'.format(qname, deltacurr)
        st += '\nInitial Search values = ({:.2f}, {:.2f})\n'.format(xini, yini)
        st += 'BBA Results:\n'
        x0s = tmp.format(xl0, stdxl0)
        y0s = tmp.format(yl0, stdyl0)
        st += '  Linear: X = {:s}  Y = {:s}\n'.format(x0s, y0s)
        x0s = tmp.format(xq0, stdxq0)
        y0s = tmp.format(yq0, stdyq0)
        st += '  Parab.: X = {:s}  Y = {:s}'.format(x0s, y0s)
        adt.text(
            0.5, 0.5, st, fontsize=10, horizontalalignment='center',
            verticalalignment='center', transform=adt.transAxes,
            bbox=dict(edgecolor='k', facecolor='w', alpha=1.0))
        adt.set_xlim([0, 8])
        adt.set_ylim([0, 8])

        rmsx = anl['quadratic_fitting']['meansqrx']
        rmsy = anl['quadratic_fitting']['meansqry']
        px = anl['quadratic_fitting']['coeffsx']
        py = anl['quadratic_fitting']['coeffsy']
        fitx = _np.polyval(px, sxpos)
        fity = _np.polyval(py, sypos)
        fitx0 = _np.polyval(px, xq0)
        fity0 = _np.polyval(py, yq0)

        aqx.plot(xpos, rmsx, 'bo')
        aqx.plot(sxpos, fitx, 'b')
        aqx.errorbar(xq0, fitx0, xerr=stdxq0, fmt='kx', markersize=20)
        aqy.plot(ypos, rmsy, 'ro')
        aqy.plot(sypos, fity, 'r')
        aqy.errorbar(yq0, fity0, xerr=stdyq0, fmt='kx', markersize=20)
        axy.errorbar(
            xq0, yq0, xerr=stdxq0, yerr=stdyq0, fmt='gx', markersize=20,
            label='parabollic')

        dorbx = anl['linear_fitting']['dorbx']
        dorby = anl['linear_fitting']['dorby']
        x0s = anl['linear_fitting']['x0s']
        y0s = anl['linear_fitting']['y0s']
        px = anl['linear_fitting']['coeffsx']
        py = anl['linear_fitting']['coeffsy']
        sidx = _np.argsort(_np.abs(px[0]))
        sidy = _np.argsort(_np.abs(py[0]))
        pvx, pvy = [], []
        npts = 6
        for ii in range(npts):
            pvx.append(_np.polyval(px[:, sidx[-ii-1]], sxpos))
            pvy.append(_np.polyval(py[:, sidy[-ii-1]], sypos))
        pvx, pvy = _np.array(pvx), _np.array(pvy)
        alx.plot(xpos, dorbx[:, sidx[-npts:]], 'b.')
        alx.plot(sxpos, pvx.T, 'b', linewidth=1)
        alx.errorbar(xl0, 0, xerr=stdxl0, fmt='kx', markersize=20)
        aly.plot(ypos, dorby[:, sidy[-npts:]], 'r.')
        aly.plot(sypos, pvy.T, 'r', linewidth=1)
        aly.errorbar(yl0, 0, xerr=stdyl0, fmt='kx', markersize=20)
        axy.errorbar(
            xl0, yl0, xerr=stdxl0, yerr=stdyl0, fmt='mx', markersize=20,
            label='linear')

        axy.legend(loc='best', fontsize='x-small')
        axy.set_xlabel(r'$X_0$ [$\mu$m]')
        axy.set_ylabel(r'$Y_0$ [$\mu$m]')
        alx.set_xlabel(r'X [$\mu$m]')
        alx.set_ylabel(r'$\Delta$ COD [$\mu$m]')
        aly.set_xlabel(r'Y [$\mu$m]')
        aly.set_ylabel(r'$\Delta$ COD [$\mu$m]')
        aqx.set_xlabel(r'X [$\mu$m]')
        aqx.set_ylabel(r'COD$^2$ [$\mu$m$^2$]')
        aqy.set_xlabel(r'Y [$\mu$m]')
        aqy.set_ylabel(r'COD$^2$ [$\mu$m$^2$]')

        if save:
            f.savefig(bpm+'.svg')
            _plt.close()
        else:
            f.show()

    def make_figure_quadfit(self, bpms=None, fname='', title=''):
        """."""
        f = _plt.figure(figsize=(9.5, 9))
        gs = _mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.78, bottom=0.15, top=0.9,
            hspace=0.5, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = _plt.subplot(gs[0, 0])
        ayy = _plt.subplot(gs[1, 0])

        bpms = bpms or self.data['bpmnames']
        colors = _cmap.brg(_np.linspace(0, 1, len(bpms)))
        for i, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                print('Data not found for ' + bpm)
                continue
            rmsx = anl['quadratic_fitting']['meansqrx']
            rmsy = anl['quadratic_fitting']['meansqry']

            px = anl['quadratic_fitting']['coeffsx']
            py = anl['quadratic_fitting']['coeffsy']

            x0 = anl['quadratic_fitting']['x0']
            y0 = anl['quadratic_fitting']['y0']

            sxpos = _np.sort(anl['xpos'])
            sypos = _np.sort(anl['ypos'])
            fitx = _np.polyval(px, sxpos)
            fity = _np.polyval(py, sypos)

            axx.plot(anl['xpos']-x0, rmsx, 'o', color=colors[i], label=bpm)
            axx.plot(sxpos-x0, fitx, color=colors[i])
            ayy.plot(anl['ypos']-y0, rmsy, 'o', color=colors[i], label=bpm)
            ayy.plot(sypos-y0, fity, color=colors[i])

        axx.legend(bbox_to_anchor=(1.0, 1.1), fontsize='xx-small')
        axx.grid(True)
        ayy.grid(True)
        axx.set_xlabel(r'$X - X_0$ [$\mu$m]')
        axx.set_ylabel(r'$\Delta$ COD$^2$')
        ayy.set_xlabel(r'$Y - Y_0$ [$\mu$m]')
        ayy.set_ylabel(r'$\Delta$ COD$^2$')
        if fname:
            f.savefig(fname+'.svg')
            _plt.close()
        else:
            f.show()

    def make_figure_linfit(self, bpms=None, fname='', title=''):
        """."""
        f = _plt.figure(figsize=(9.5, 9))
        gs = _mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.78, bottom=0.15, top=0.9,
            hspace=0.5, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = _plt.subplot(gs[0, 0])
        axy = _plt.subplot(gs[1, 0])

        bpms = bpms or self.data['bpmnames']
        colors = _cmap.brg(_np.linspace(0, 1, len(bpms)))
        for i, bpm in enumerate(bpms):
            anl = self.analysis.get(bpm)
            if not anl:
                print('Data not found for ' + bpm)
                continue
            x0 = anl['linear_fitting']['x0']
            y0 = anl['linear_fitting']['y0']
            px = anl['linear_fitting']['coeffsx']
            py = anl['linear_fitting']['coeffsy']

            sidx = _np.argsort(_np.abs(px[0]))
            sidy = _np.argsort(_np.abs(py[0]))

            xpos = anl['xpos']
            ypos = anl['ypos']
            sxpos = _np.sort(xpos)
            sypos = _np.sort(ypos)

            pvx, pvy = [], []
            for ii in range(3):
                pvx.append(_np.polyval(px[:, sidx[ii]], sxpos))
                pvy.append(_np.polyval(py[:, sidy[ii]], sypos))
            pvx, pvy = _np.array(pvx), _np.array(pvy)

            axx.plot(sxpos, pvx.T, color=colors[i])
            axx.plot(x0, 0, 'x', markersize=20, color=colors[i], label=bpm)
            axy.plot(sypos, pvy.T, color=colors[i])
            axy.plot(y0, 0, 'x', markersize=20, color=colors[i], label=bpm)

        axx.legend(
            loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='xx-small')
        axx.grid(True)
        axy.grid(True)

        if fname:
            f.savefig(fname+'.svg')
            _plt.close()
        else:
            f.show()

    def make_figure_compare_with_initial(
            self, method='linear_fitting', bpmsok=None, bpmsnok=None,
            xlim=None, ylim=None, fname='', title='', plotdiff=True):
        """."""
        f = _plt.figure(figsize=(9.2, 9))
        gs = _mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.98, bottom=0.08, top=0.9,
            hspace=0.01, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = _plt.subplot(gs[0, 0])
        ayy = _plt.subplot(gs[1, 0], sharex=axx)

        bpmsok = bpmsok or self.data['bpmnames']
        bpmsnok = bpmsnok or []
        iok = _np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsok], dtype=int)
        inok = _np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsnok], dtype=int)

        labels = ['initial', method]
        cors = _cmap.brg(_np.linspace(0, 1, 3))

        x0c = _np.array(self.data['scancenterx'])
        y0c = _np.array(self.data['scancentery'])
        x0q, y0q, stdx0q, stdy0q = self.get_bba_results(
            method=method, error=True)
        if plotdiff:
            x0q -= x0c
            y0q -= y0c
            x0c -= x0c
            y0c -= y0c

        minx = _np.min(
            _np.hstack([x0q[iok], x0c[iok], x0q[inok], x0c[inok]]))*1.1
        maxx = _np.max(
            _np.hstack([x0q[iok], x0c[iok], x0q[inok], x0c[inok]]))*1.1
        miny = _np.min(
            _np.hstack([y0q[iok], y0c[iok], y0q[inok], y0c[inok]]))*1.1
        maxy = _np.max(
            _np.hstack([y0q[iok], y0c[iok], y0q[inok], y0c[inok]]))*1.1
        minx = -1*xlim if xlim is not None else minx
        maxx = xlim if xlim is not None else maxx
        miny = -1*ylim if ylim is not None else miny
        maxy = ylim if ylim is not None else maxy

        axx.errorbar(iok, x0c[iok], fmt='o', color=cors[0], label=labels[0])
        axx.errorbar(
            iok, x0q[iok], yerr=stdx0q[iok], fmt='o', color=cors[1],
            label=labels[1], elinewidth=1)
        ayy.errorbar(iok, y0c[iok], fmt='o', color=cors[0])
        ayy.errorbar(
            iok, y0q[iok], yerr=stdy0q[iok], fmt='o', color=cors[1],
            elinewidth=1)

        if inok.size:
            axx.errorbar(inok, x0c[inok], fmt='x', color=cors[0])
            axx.errorbar(
                inok, x0q[inok], yerr=stdx0q[inok], fmt='x', color=cors[1],
                elinewidth=1)
            ayy.errorbar(
                inok, y0c[inok], fmt='x', color=cors[0], label=labels[0])
            ayy.errorbar(
                inok, y0q[inok], yerr=stdy0q[inok], fmt='x', color=cors[1],
                elinewidth=1, label=labels[1])

        axx.legend(
            loc='lower right', bbox_to_anchor=(1, 1), fontsize='small', ncol=2)
        axx.grid(True)
        ayy.grid(True)
        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])

        if plotdiff:
            axx.set_ylabel(r'$\Delta X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$\Delta Y_0$ [$\mu$m]')
        else:
            axx.set_ylabel(r'$X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$Y_0$ [$\mu$m]')
        ayy.set_xlabel('BPM Index')

        if fname:
            f.savefig(fname+'.svg')
            _plt.close()
        else:
            f.show()

    def make_figure_compare_methods(
            self, bpmsok=None, bpmsnok=None, xlim=None, ylim=None, fname='',
            title='', plotdiff=True):
        """."""
        f = _plt.figure(figsize=(9.2, 9))
        gs = _mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.1, right=0.98, bottom=0.08, top=0.9,
            hspace=0.01, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = _plt.subplot(gs[0, 0])
        ayy = _plt.subplot(gs[1, 0], sharex=axx)

        bpmsok = bpmsok or self.data['bpmnames']
        bpmsnok = bpmsnok or []
        iok = _np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsok], dtype=int)
        inok = _np.array(
            [self.data['bpmnames'].index(bpm) for bpm in bpmsnok], dtype=int)

        labels = ['linear', 'quadratic']
        cors = _cmap.brg(_np.linspace(0, 1, 3))

        x0l, y0l, stdx0l, stdy0l = self.get_bba_results(
            method='linear_fitting', error=True)
        x0q, y0q, stdx0q, stdy0q = self.get_bba_results(
            method='quadratic_fitting', error=True)
        if plotdiff:
            x0q -= x0l
            y0q -= y0l
            x0l -= x0l
            y0l -= y0l

        minx = _np.min(
            _np.hstack([x0q[iok], x0l[iok], x0q[inok], x0l[inok]]))*1.1
        maxx = _np.max(
            _np.hstack([x0q[iok], x0l[iok], x0q[inok], x0l[inok]]))*1.1
        miny = _np.min(
            _np.hstack([y0q[iok], y0l[iok], y0q[inok], y0l[inok]]))*1.1
        maxy = _np.max(
            _np.hstack([y0q[iok], y0l[iok], y0q[inok], y0l[inok]]))*1.1
        minx = -1*xlim if xlim is not None else minx
        maxx = xlim if xlim is not None else maxx
        miny = -1*ylim if ylim is not None else miny
        maxy = ylim if ylim is not None else maxy

        axx.errorbar(
            iok, x0l[iok], yerr=stdx0l[iok], fmt='o', color=cors[0],
            label=labels[0])
        axx.errorbar(
            iok, x0q[iok], yerr=stdx0q[iok], fmt='o', color=cors[1],
            label=labels[1], elinewidth=1)
        ayy.errorbar(
            iok, y0l[iok], yerr=stdy0l[iok], fmt='o', color=cors[0])
        ayy.errorbar(
            iok, y0q[iok], yerr=stdy0q[iok], fmt='o', color=cors[1],
            elinewidth=1)

        if inok.size:
            axx.errorbar(
                inok, x0l[inok], yerr=stdx0l[inok], fmt='x', color=cors[0])
            axx.errorbar(
                inok, x0q[inok], yerr=stdx0q[inok], fmt='x', color=cors[1],
                elinewidth=1,)
            ayy.errorbar(
                inok, y0l[inok], yerr=stdy0l[inok], fmt='x', color=cors[0],
                label=labels[0])
            ayy.errorbar(
                inok, y0q[inok], yerr=stdy0q[inok], fmt='x', color=cors[1],
                elinewidth=1, label=labels[1])

        axx.legend(
            loc='lower right', bbox_to_anchor=(1, 1), fontsize='small',
            ncol=2, title='Fitting method')
        axx.grid(True)
        ayy.grid(True)
        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])

        if plotdiff:
            axx.set_ylabel(r'$\Delta X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$\Delta Y_0$ [$\mu$m]')
        else:
            axx.set_ylabel(r'$X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$Y_0$ [$\mu$m]')
        ayy.set_xlabel('BPM Index')

        if fname:
            f.savefig(fname+'.svg')
            _plt.close()
        else:
            f.show()

    @staticmethod
    def make_figure_compare_bbas(
            bbalist, method='linear_fitting', labels=None, bpmsok=None,
            bpmsnok=None, fname='', xlim=None, ylim=None, title='',
            plotdiff=True):
        """."""
        f = _plt.figure(figsize=(9.2, 9))
        gs = _mpl_gs.GridSpec(2, 1)
        gs.update(
            left=0.12, right=0.98, bottom=0.13, top=0.9, hspace=0, wspace=0.35)

        if title:
            f.suptitle(title)

        axx = _plt.subplot(gs[0, 0])
        ayy = _plt.subplot(gs[1, 0], sharex=axx)

        bpmsok = bpmsok or bbalist[0].data['bpmnames']
        bpmsnok = bpmsnok or []
        iok = _np.array(
            [bbalist[0].data['bpmnames'].index(bpm) for bpm in bpmsok],
            dtype=int)
        inok = _np.array(
            [bbalist[0].data['bpmnames'].index(bpm) for bpm in bpmsnok],
            dtype=int)

        if labels is None:
            labels = [str(i) for i in range(len(bbalist))]
        cors = _cmap.brg(_np.linspace(0, 1, len(bbalist)))

        minx = miny = _np.inf
        maxx = maxy = -_np.inf
        x0li, y0li, = bbalist[0].get_bba_results(
            method=method, error=False)
        for i, dobba in enumerate(bbalist):
            x0l, y0l, stdx0l, stdy0l = dobba.get_bba_results(
                method=method, error=True)
            if plotdiff:
                x0l -= x0li
                y0l -= y0li

            minx = _np.min(_np.hstack([minx, x0l[iok], x0l[inok]]))
            maxx = _np.max(_np.hstack([maxx, x0l[iok], x0l[inok]]))
            miny = _np.min(_np.hstack([miny, y0l[iok], y0l[inok]]))
            maxy = _np.max(_np.hstack([maxy, y0l[iok], y0l[inok]]))

            axx.errorbar(
                iok, x0l[iok], yerr=stdx0l[iok], fmt='o', color=cors[i],
                label=labels[i])
            ayy.errorbar(
                iok, y0l[iok], yerr=stdy0l[iok], fmt='o', color=cors[i],
                elinewidth=1)

            if not inok.size:
                continue

            axx.errorbar(
                inok, x0l[inok], yerr=stdx0l[inok], fmt='x', color=cors[i])
            ayy.errorbar(
                inok, y0l[inok], yerr=stdy0l[inok], fmt='x', color=cors[i],
                elinewidth=1, label=labels[i])

        if inok.size:
            ayy.legend(
                loc='upper right', bbox_to_anchor=(1.8, 0.2),
                fontsize='xx-small')

        axx.legend(
            loc='lower right', bbox_to_anchor=(1, 1), fontsize='xx-small')
        axx.grid(True)
        ayy.grid(True)

        minx = -1*xlim if xlim else minx*1.1
        maxx = xlim if xlim else maxx*1.1
        miny = -1*ylim if ylim else miny*1.1
        maxy = ylim if ylim else maxy*1.1

        axx.set_ylim([minx, maxx])
        ayy.set_ylim([miny, maxy])
        ayy.set_xlabel('BPM Index')

        if plotdiff:
            axx.set_ylabel(r'$\Delta X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$\Delta Y_0$ [$\mu$m]')
        else:
            axx.set_ylabel(r'$X_0$ [$\mu$m]')
            ayy.set_ylabel(r'$Y_0$ [$\mu$m]')

        if fname:
            f.savefig(fname+'.svg')
            _plt.close()
        else:
            f.show()

    # #### private methods ####

    def _ok_to_continue(self):
        if self._stopevt.is_set():
            print("stopped!")
            return False
        if not self.devices['dcct'].havebeam:
            print("Beam lost!")
            return False
        return True

    def _meas_bba(self):
        tini = _datetime.datetime.fromtimestamp(_time.time())
        print('Starting measurement at {:s}'.format(
            tini.strftime('%Y-%m-%d %Hh%Mm%Ss')))

        sofb = self.devices['sofb']
        sofb.orb_nrpoints = self.params.sofb_nrpoints

        for i, bpm in enumerate(self._bpms2dobba):
            if not self._ok_to_continue():
                break
            print('\nCorrecting Orbit...', end='')
            self.correct_orbit()
            print('Ok!')
            print('\n{0:03d}/{1:03d}'.format(i+1, len(self._bpms2dobba)))
            self._meas_bba_single_bpm(bpm)

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split('.')[0]
        print('finished! Elapsed time {:s}'.format(dtime))

    def _meas_bba_single_bpm(self, bpmname):
        """."""
        idx = self.data['bpmnames'].index(bpmname)

        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime('%Hh%Mm%Ss')
        print('{:s} --> Doing BBA for BPM {:03d}: {:s}'.format(
            strtini, idx, bpmname))

        quadname = self.data['quadnames'][idx]
        x0 = self.data['scancenterx'][idx]
        y0 = self.data['scancentery'][idx]
        quad = self.devices[quadname]
        sofb = self.devices['sofb']

        if not quad.pwrstate:
            print('\n    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        curr0 = quad.current
        deltacurr = self.params.quad_deltacurr
        cycling_curve = BBA.get_cycling_curve()

        upp = self.params.quad_maxcurr
        low = self.params.quad_mincurr
        # Limits are interchanged in some quads:
        upplim = max(upp, low) - 0.0005
        lowlim = min(upp, low) + 0.0005

        print('cycling ' + quadname + ': ', end='')
        for _ in range(self.params.quad_nrcycles):
            print('.', end='')
            for fac in cycling_curve:
                newcurr = min(max(curr0 + deltacurr*fac, lowlim), upplim)
                quad.current = newcurr
                _time.sleep(self.params.wait_quadrupole)
        print(' Ok!')

        nrsteps = self.params.meas_nrsteps
        dorbsx = self._calc_dorb_scan(self.params.deltaorbx, nrsteps//2)
        dorbsy = self._calc_dorb_scan(self.params.deltaorby, nrsteps//2)

        refx0, refy0 = sofb.ref_orbx.copy(), sofb.ref_orby.copy()
        enblx0, enbly0 = sofb.bpmx_enbl.copy(), sofb.bpmy_enbl.copy()
        ch0, cv0 = sofb.currents_hcm.copy(), sofb.currents_vcm.copy()

        enblx, enbly = 0*enblx0, 0*enbly0
        enblx[idx], enbly[idx] = 1, 1
        sofb.bpmx_enbl, sofb.bpmy_enbl = enblx, enbly

        orbini, orbpos, orbneg = [], [], []
        npts = 2*(nrsteps//2) + 1
        tmpl = '{:25s}'.format
        currpos = currneg = 0.0
        for i in range(npts):
            if self._stopevt.is_set() or not self.devices['dcct'].havebeam:
                print('   exiting...')
                break
            print('    {0:02d}/{1:02d} --> '.format(i+1, npts), end='')

            print('orbit corr: ', end='')
            ret, fmet = self.correct_orbit_at_bpm(
                bpmname, x0+dorbsx[i], y0+dorbsy[i])
            if fmet <= self.params.sofb_maxorberr:
                txt = tmpl('Ok! in {:02d} iters'.format(ret))
            else:
                txt = tmpl('NOT Ok! dorb={:5.1f} um'.format(fmet))
            print(txt, end='')

            orbini.append(sofb.get_orbit())

            for j, fac in enumerate(cycling_curve):
                newcurr = min(max(curr0 + deltacurr*fac, lowlim), upplim)
                quad.current = newcurr
                _time.sleep(self.params.wait_quadrupole)
                if j == 0:
                    orbpos.append(sofb.get_orbit())
                    currpos = quad.current
                elif j == 1:
                    orbneg.append(sofb.get_orbit())
                    currneg = quad.current

            dorb = orbpos[-1] - orbneg[-1]
            dorbx = dorb[:len(self.data['bpmnames'])]
            dorby = dorb[len(self.data['bpmnames']):]
            rmsx = _np.sqrt(_np.sum(dorbx*dorbx) / dorbx.shape[0])
            rmsy = _np.sqrt(_np.sum(dorby*dorby) / dorby.shape[0])
            print('rmsx = {:5.1f} rmsy = {:5.1f} um dcurr = {:.1g}'.format(
                rmsx, rmsy, currpos - currneg))

        self.data['measure'][bpmname] = {
            'orbini': _np.array(orbini),
            'orbpos': _np.array(orbpos),
            'orbneg': _np.array(orbneg),
            'currpos': currpos,
            'currneg': currneg}

        print('    restoring initial conditions.')
        sofb.ref_orbx, sofb.ref_orby = refx0, refy0
        sofb.bpmx_enbl, sofb.bpmy_enbl = enblx0, enbly0

        # restore correctors gently to do not kill the beam.
        factch, factcv = sofb.corr_gain_hcm, sofb.corr_gain_vcm
        chn, cvn = sofb.currents_hcm, sofb.currents_vcm
        dch, dcv = ch0 - chn, cv0 - cvn
        sofb.delta_currents_hcm, sofb.delta_currents_vcm = dch, dcv
        nrsteps = _np.ceil(max(_np.abs(dch).max(), _np.abs(dcv).max()) / 1.0)
        for i in range(int(nrsteps)):
            sofb.corr_gain_hcm = (i+1)/nrsteps * 100
            sofb.corr_gain_vcm = (i+1)/nrsteps * 100
            sofb.apply_correction()
            _time.sleep(self.params.wait_correctors)
        sofb.delta_currents_hcm, sofb.delta_currents_vcm = dch*0, dcv*0
        sofb.corr_gain_hcm, sofb.corr_gain_vcm = factch, factcv

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split('.')[0]
        print('Done! Elapsed time: {:s}\n'.format(dtime))
