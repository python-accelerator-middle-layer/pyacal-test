"""Coupling Measurement from Minimal Tune Separation."""

import numpy as _np
import matplotlib.pyplot as _plt


from ..optimization import SimulAnneal as _SimulAnneal


class FitTunes(_SimulAnneal):
    """Tune Fit.

    tunex = coeff1 * quad_parameter + offset1
    tuney = coeff2 * quad_parameter + offset2

    tune1, tune2 = Eigenvalues([[tunex, coupling/2], [coupling/2, tuney]])

    fit parameters: coeff1, offset1, coeff2, offset2, coupling

    Ex.:
            fittune = FitTunes(
                param=quad_strengths, tune1=meas_tune1, tune2=meas_tune2)
            fittune.fitting_plot()
            fittune.niter = 1000
            fittune.start()
            fittune.fitting_plot()

    NOTE: It maybe necessary to add a quadratic quadrupole strength
          dependency for tunes!
    """

    def __init__(self, param, tune1, tune2):
        """."""
        self._param = _np.asarray(param)
        self._tune1 = _np.asarray(tune1)
        self._tune2 = _np.asarray(tune2)

        # sort tune1 > tune2 at each point
        sel = self._tune1 <= self._tune2
        if sel.any():
            self._param = _np.array(param)
            self._tune1 = _np.array(tune1)
            self._tune2 = _np.array(tune2)
            self._tune1[sel], self._tune2[sel] = \
                self._tune2[sel], self._tune1[sel]

        super().__init__(use_thread=False)

    # --- fitting parameters ---

    @property
    def param(self):
        """."""
        return self._param

    @property
    def tune1(self):
        """."""
        return self._tune1

    @property
    def tune2(self):
        """."""
        return self._tune2

    @property
    def coeff1(self):
        """."""
        return self.position[0]

    @coeff1.setter
    def coeff1(self, value):
        """."""
        self.position[0] = value

    @property
    def offset1(self):
        """."""
        return self.position[1]

    @offset1.setter
    def offset1(self, value):
        """."""
        self.position[1] = value

    @property
    def coeff2(self):
        """."""
        return self.position[2]

    @coeff2.setter
    def coeff2(self, value):
        """."""
        self.position[2] = value

    @property
    def offset2(self):
        """."""
        return self.position[3]

    @offset2.setter
    def offset2(self, value):
        """."""
        self.position[3] = value

    @property
    def coupling(self):
        """."""
        return self.position[4]

    @coupling.setter
    def coupling(self, value):
        """."""
        self.position[4] = value

    def calc_obj_fun(self):
        """."""
        # print(self.position)
        coeff1, offset1, coeff2, offset2, coupling = self.position
        _, tune1, tune2 = \
            FitTunes.calc_tunes(
                self._param, coeff1, offset1, coeff2, offset2, coupling)
        diff = ((tune1 - self._tune1)**2 + (tune2 - self._tune2)**2)**0.5
        residue = _np.mean(diff)
        return residue

    def initialization(self):
        """."""
        self._calc_init_parms()
        coeff1, offset1, coeff2, offset2, _ = self.position
        self.deltas = _np.array([
            0.002 * (coeff1 - coeff2),
            0.002 * (offset1 - offset2),
            0.002 * (coeff1 - coeff2),
            0.002 * (offset1 - offset2),
            0.5 * 0.1/100])

    @staticmethod
    def calc_tunes(
            param=None,
            coeff1=None, offset1=None,
            coeff2=None, offset2=None, coupling=None):
        """."""
        fx_ = coeff1 * param + offset1
        fy_ = coeff2 * param + offset2
        tune1, tune2 = _np.zeros(param.shape), _np.zeros(param.shape)
        for i in range(len(param)):
            mat = _np.array([[fx_[i], coupling/2], [coupling/2, fy_[i]]])
            eigvalues, _ = _np.linalg.eig(mat)
            tune1[i], tune2[i] = eigvalues
            if tune1[i] <= tune2[i]:
                tune1[i], tune2[i] = tune2[i], tune1[i]
        return param, tune1, tune2

    def fitting_plot(self, title=None, xlabel=None, fig=None):
        """."""
        position = self.position

        # NOTE: do error estimate properly!
        coup_error = self.calc_obj_fun()

        if xlabel is None:
            xlabel = 'Quadrupole Integrated Strength [1/m]'
        if title is None:
            title = 'Transverse Linear Coupling: {:.2f} ± {:.2f}%'.format(
                position[-1] * 100, coup_error * 100)

        prop_cycle = _plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        if fig is None:
            fig = _plt.figure()
        param, fittune1, fittune2 = FitTunes.calc_tunes(self._param, *position)

        # plot meas data
        _plt.plot(
            self._param, self._tune1, 'o', color=colors[0],
            label='measurement')
        _plt.plot(self._param, self._tune2, 'o', color=colors[0])

        # plot fitting
        _plt.plot(param, fittune1, color=colors[1], label='fitting')
        _plt.plot(param, fittune2, color=colors[1])
        _plt.legend()
        _plt.xlabel(xlabel)
        _plt.ylabel('Transverse Tunes')
        _plt.title(title)
        _plt.grid()
        _plt.show()

    def _calc_init_parms(self):
        """."""
        parm = self._param
        vyb = self._tune1[0], self._tune2[0]
        vye = self._tune1[-1], self._tune2[-1]
        dparm = parm[-1] - parm[0]
        coeff1 = (min(vye) - max(vyb)) / dparm
        offset1 = max(vyb) - coeff1 * parm[0]
        coeff2 = (max(vye) - min(vyb)) / dparm
        offset2 = min(vyb) - coeff2 * parm[0]
        coupling = min(_np.abs(self._tune1 - self._tune2))
        position = [coeff1, offset1, coeff2, offset2, coupling]
        self.position = position
