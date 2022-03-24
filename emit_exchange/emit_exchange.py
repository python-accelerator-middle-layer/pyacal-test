import numpy as _np
import pyaccel as _pa
import pymodels as _pm


class EmittanceExchangeSimul:

    def __init__(self, accelerator='bo'):

        ACC_LIST = ['bo', ]

        self._model = None

        if accelerator not in ACC_LIST:
            raise NotImplementedError(
                'Simulation not implemented for the passed accelerator')

        self._model = _pm.bo.create_accelerator(energy=3e9)

    @property
    def model(self):
        return self._model

    @staticmethod
    def calc_emit_exchange_quality(self, emit1_0, emit2_0, emit1):
        """."""
        r = 1 - (emit1 - emit2_0)/(emit1_0 - emit2_0)
        return r

    @staticmethod
    def C_to_KsL(self, C):
        fam_data = _pm.bo.get_family_data(self.model)
        qs_idx = fam_data['QS']['index']
        ed_tang, *_ = _pa.optics.calc_edwards_teng(accelerator=self.model)
        beta1 = ed_tang.beta1[qs_idx[0]]
        beta2 = ed_tang.beta2[qs_idx[0]]
        KsL = -2 * _np.pi * C / _np.sqrt(beta1 * beta2)

        return KsL[0]

    @staticmethod
    def KsL_to_C(self, KsL):
        fam_data = _pm.bo.get_family_data(self.model)
        qs_idx = fam_data['QS']['index']
        ed_tang, *_ = _pa.optics.calc_edwards_teng(accelerator=self.model)
        beta1 = ed_tang.beta1[qs_idx[0][0]]
        beta2 = ed_tang.beta2[qs_idx[0][0]]
        C = _np.abs(KsL * _np.sqrt(beta1 * beta2)/(2 * _np.pi))

        return C
