"""."""
from pymodels import si
from siriuspy.clientconfigdb import ConfigDBClient

from ..utils import ThreadedMeasBaseClass as _BaseClass

from .measure_beta import MeasBeta as _MeasBeta
from .measure_respmat import MeasureRespMat as _MeasureRespMat


class RespMatBetaParams:
    """."""

    def __init__(self):
        """."""
        self.nmeas_respmat = 10
        self.nmeas_beta = 10
        self.filename = 'variation'

    def __str__(self):
        """."""
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nmeas_beta', self.nmeas_beta, '')
        stg += dtmp('nmeas_respmat', self.nmeas_respmat, '')
        stg += f"filename = '{self.filename:s}'\n"
        return stg


class MeasureRespMatBeta(_BaseClass):
    """."""

    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(self):
        """."""
        super().__init__(params=RespMatBetaParams(), target=self._run_all)
        self.meas_beta = self._get_measbeta_object()
        self.meas_respmat = _MeasureRespMat()
        self.confdb = ConfigDBClient(config_type='si_orbcorr_respm')

    def __str__(self):
        """."""
        stg = str(self.params)
        stg += '\n\n MeasRespMat Params:\n\n'
        stg += str(self.meas_respmat.params)
        stg += '\n\n MeasBeta Params:\n\n'
        stg += str(self.meas_beta.params)
        return stg

    @property
    def connected(self):
        """."""
        conn = super().connected
        conn &= self.meas_respmat.connected
        conn &= self.meas_beta.connected
        return conn

    def _run_all(self):
        """."""
        self._run_respmat()
        self._run_beta()

    def _run_respmat(self):
        """."""
        for nms in range(self.params.nmeas_respmat):
            if self._stopevt.is_set():
                break
            print('Respmat Measurement Number {:d}'.format(nms+1))
            respmat_name = f'{self.params.filename:s}_n{nms+1:02d}'
            self.meas_respmat.params.respmat_name = respmat_name
            self.meas_respmat.start()
            self.meas_respmat.wait_measurement()
            self.meas_respmat.save_data('respmat_' + respmat_name)

    def _run_beta(self):
        """."""
        for nms in range(self.params.nmeas_beta):
            if self._stopevt.is_set():
                break
            print('Beta Measurement Number {:d}'.format(nms+1))
            # correct orbit:
            sofb = self.meas_respmat.devices['sofb']
            sofb.correct_orbit_manually(self.meas_respmat.params.corr_nr_iters)

            # start measurement:
            beta_name = f'beta_{self.params.filename:s}_n{nms+1:02d}'
            self.meas_beta = self._get_measbeta_object()
            if self._stopevt.is_set():
                break
            self.meas_beta.start()
            self.meas_beta.wait_measurement()
            self.meas_beta.process_data(mode='pos')
            self.meas_beta.save_data(beta_name)

    def _get_measbeta_object(self):
        """."""
        model = si.create_accelerator()
        model.cavity_on = True
        famdata = si.get_family_data(model)
        measbeta = _MeasBeta(model, famdata)
        measbeta.params.nr_measures = 1
        measbeta.params.quad_deltakl = 0.01/2  # [1/m]
        measbeta.params.wait_quadrupole = 1  # [s]
        measbeta.params.wait_tune = 1  # [s]
        measbeta.params.timeout_quad_turnon = 10  # [s]
        measbeta.params.recover_tune = True
        measbeta.params.recover_tune_tol = 1e-4
        measbeta.params.recover_tune_maxiter = 5

        measbeta.params.quad_nrcycles = 0
        measbeta.params.time_wait_quad_cycle = 0.3  # [s]
        measbeta.quads2meas = list(measbeta.data['quadnames'])
        return measbeta
