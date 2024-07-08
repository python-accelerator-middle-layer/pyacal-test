"""."""

import operator as _operator

import numpy as _np
from scipy.constants import speed_of_light as _speed_of_light

from .. import _get_facility
from .utils import ConverterTypes

__CONVERTERS_DICT = {}
_PV_class = None


def create_converter(converter_type, kwargs):
    if converter_type not in ConverterTypes:
        raise TypeError(f'Converter type {converter_type} is not allowed.')

    clss = getattr(globals(), converter_type)
    key = (converter_type, clss.get_key(**kwargs))
    conv = __CONVERTERS_DICT.get(key)
    if conv is None:
        __CONVERTERS_DICT[key] = clss(**kwargs)
    return __CONVERTERS_DICT[key]


def get_converter(key):
    conv = __CONVERTERS_DICT.get(key)
    if conv is None:
        raise KeyError(
            'Converter not created yet. use `create_converter` first.'
        )
    return conv


class _BaseConverter:

    def __init__(self):
        pass

    @staticmethod
    def get_key():
        return ()

    def conversion_forward(self, value):
        return value

    def conversion_reverse(self, value):
        return value


class ScaleConverter(_BaseConverter):

    def __init__(self, scale=1.0):
        super().__init__()
        if not isinstance(scale, (int, float, _np.int64, _np.float64)):
            raise TypeError('Scale must be a number.')
        scale = float(scale)
        if _np.isnan(scale) or _np.isinf(scale) or scale == 0.0:
            raise TypeError('Scale must not be inf, nan or 0.')
        self._scale = float(scale)

    @staticmethod
    def get_key(scale=1.0):
        return (scale, )

    def conversion_forward(self, value):
        return value * self._scale

    def conversion_reverse(self, value):
        return value / self._scale


class OffsetConverter(_BaseConverter):

    def __init__(self, offset=1.0):
        super().__init__()
        if not isinstance(offset, (int, float, _np.int64, _np.float64)):
            raise TypeError('Offset must be a number.')
        offset = float(offset)
        if _np.isnan(offset) or _np.isinf(offset):
            raise TypeError('Offset must not be inf or nan.')
        self._offset = float(offset)

    @staticmethod
    def get_key(offset=1.0):
        return (offset, )

    def conversion_forward(self, value):
        return value + self._offset

    def conversion_reverse(self, value):
        return value - self._offset


class LookupTableConverter(_BaseConverter):

    def __init__(self, table_name=''):
        super().__init__()

        facil = _get_facility()
        table = _np.array(facil.get_lookup_table(table_name))

        if table.dtype not in (_np.float64, _np.int64):
            raise ValueError('Lookup table must be of dtype int or float.')
        elif table.ndim != 2 or table.shape[1] != 2:
            raise ValueError('Lookup table must have shape (N, 2).')

        diff = _np.diff(table, axis=0)
        diff *= diff[0]
        if _np.any(diff <= 0.0):
            raise ValueError(
                'Lookup table must be strictly monotonic to be invertible.'
            )
        self._table = table

    @staticmethod
    def get_key(table_name=''):
        return (table_name, )

    def conversion_forward(self, value):
        val = _np.interp(
            value,
            self.table[:, 0],
            self.table[:, 1],
            left=_np.nan,
            right=_np.nan
        )
        if _np.isnan(val):
            raise ValueError(f'Value {value} is out of bounds.')
        return val

    def conversion_reverse(self, value):
        val = _np.interp(
            value,
            self.table[:, 1],
            self.table[:, 0],
            left=_np.nan,
            right=_np.nan
        )
        if _np.isnan(val):
            raise ValueError(f'Value {value} is out of bounds.')
        return val


class PolynomialConverter(_BaseConverter):

    def __init__(self, coeffs=(0, 1), limits=(0, 1)):
        super().__init__()

        if not isinstance(limits, (list, tuple, _np.ndarray)):
            raise TypeError('Limits must be list, tuple or 1d numpy.ndarray.')
        elif not isinstance(coeffs, (list, tuple, _np.ndarray)):
            raise TypeError('Coeffs must be list, tuple or 1d numpy.ndarray.')

        pol = _np.polynomial.Polynomial(coeffs, domain=tuple(limits))
        pol_der = pol.deriv()
        roots = pol_der.roots()
        roots = roots[_np.isreal(roots)]
        if _np.any(roots > limits[0]) and _np.any(roots < limits[1]):
            raise TypeError(
                'Polynom must be strictly monotonic to be invertible.'
            )
        self._pol = pol
        self._pol_der = pol_der

        x = _np.linspace(*limits)
        y = pol(x)
        if y[-1] < y[0]:
            y = y[::-1]
            x = x[::-1]
        self._table_guess = _np.vstack([y, x]).T

    @staticmethod
    def get_key(coefs=(0, 1), limits=(0, 1)):
        return (tuple(coefs), tuple(limits))

    def conversion_forward(self, value):
        if not self._pol.domain[0] <= value <= self._pol.domain[1]:
            raise ValueError(f'Value {value} is out of bounds.')
        return self._pol(value)

    def conversion_reverse(self, value):
        val = _np.interp(
            value,
            self._table_guess[:, 0],
            self._table_guess[:, 1],
            left=_np.nan,
            right=_np.nan
        )
        if _np.isnan(val):
            raise ValueError(f'Value {value} is out of bounds.')

        res = _np.finfo(_np.float64).resolution
        for _ in range(10):
            dval = (value - self._pol(val)) / self._pol_der(val)
            val += dval
            if _np.isclose(dval, 0, rtol=0, atol=res):
                break
        return val


class CompanionProptyConverter(_BaseConverter):

    _ADD_SUB = {'add', 'sub'}
    _MUL_DIV = {'mul', 'div'}

    def __init__(self, devname='', propty='', operation='add'):
        global _PV_class
        if _PV_class is None:
            from .pv import PV
            _PV_class = PV
        self._prpt_pv = _PV_class(devname, propty, auto_monitor=True)

        if operation in self._ADD_SUB:
            opr_inv = self._ADD_SUB - {operation, }
        elif operation in self._MUL_DIV:
            opr_inv = self._MUL_DIV - {operation, }
        else:
            raise ValueError('Operation must be "add", "sub", "mul" or "div".')

        opr_inv = opr_inv.pop()
        self._opr_fwd = 'truediv' if operation == 'div' else operation
        self._opr_inv = 'truediv' if opr_inv == 'div' else opr_inv

    @staticmethod
    def get_key(devname='', propty='', operation='add'):
        return (devname, propty, operation)

    def conversion_forward(self, value):
        val_comp = self._get_companion_property_value()
        return getattr(_operator, self._opr_fwd)(value, val_comp)

    def conversion_reverse(self, value):
        val_comp = self._get_companion_property_value()
        return getattr(_operator, self._opr_inv)(value, val_comp)

    def _get_companion_property_value(self):
        if not self._prpt_pv.connected:
            raise ValueError('Companion property not connected.')
        val_comp = self._prpt_pv.value
        if val_comp is None:
            raise ValueError('Companion property value is None.')
        return val_comp


class MagRigidityConverter(CompanionProptyConverter):
    """Ultra-relativistic approximation is used here."""

    _CONST = _speed_of_light  # c / (E/e)

    def __init__(self, devname='', propty='', energy=0, conv_2_ev=1e9):
        self._const = self._CONST / conv_2_ev
        self._energy = energy
        if not energy:
            super().__init__(devname=devname, propty=propty, operation='div')
        else:
            self._const /= energy

    @staticmethod
    def get_key(devname='', propty='', energy=0, conv_2_ev=1e9):
        return (devname, propty, energy, conv_2_ev)

    def conversion_forward(self, value):
        if not self._energy:
            value = super().conversion_forward(value)
        return value * self._const

    def conversion_reverse(self, value):
        if not self._energy:
            value = super().conversion_reverse(value)
        return value / self._const
