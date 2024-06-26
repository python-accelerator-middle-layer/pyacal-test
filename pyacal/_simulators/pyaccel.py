import numpy as _np
import pyaccel as pa

from .. import FACILITY


def get_positions(indices, acc=None):
    """Return the longitudinal position along the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the elements
            where to return data;
        acc (str, optional): name of the accelerator. Defaults to None, which
            means the variable DEFAULT_ACCELERATOR will be used.

    Returns:
        numpy.ndarray: position in [m] along the accelerator.
    """
    accel = FACILITY.accelerators[acc]
    return _np.array(pa.lattice.find_spos(accel, indices=indices))


def get_orbit(indices, acc=None):
    """Return the orbit of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the elements
            where to return data;
        acc (str, optional): name of the accelerator. Defaults to None, which
            means the variable DEFAULT_ACCELERATOR will be used.

    Returns:
        tuple: (orbx, orby) in [m].
    """
    accel = FACILITY.accelerators[acc]
    orb = pa.tracking.find_orbit6(accel, indices=indices)
    return orb[0], orb[2]


def get_twiss(indices, acc=None):
    """Return twiss parameters of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the elements
            where to return data;
        acc (str, optional): name of the accelerator. Defaults to None, which
            means the variable DEFAULT_ACCELERATOR will be used.

    Returns:
        dict: Dictionary containing Twiss parameters. Available keys:
            'betax', 'betay', 'alphax', 'alphay', 'mux', 'muy',
            'etax', 'etay', 'etapx', 'etapy'.
    """
    accel = FACILITY.accelerators[acc]
    twi, *_ = pa.optics.calc_twiss(accel, indices=indices)
    return {
        'betax': twi.betax,
        'betay': twi.betay,
        'alphax': twi.alphax,
        'alphay': twi.alphay,
        'mux': twi.mux,
        'muy': twi.muy,
        'etax': twi.etax,
        'etay': twi.etay,
        'etapx': twi.etapx,
        'etapy': twi.etapy,
    }


def get_beamsizes(indices, acc=None):
    """Return beam sizes of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the elements
            where to return data;
        acc (str, optional): name of the accelerator. Defaults to None, which
            means the variable DEFAULT_ACCELERATOR will be used.

    Returns:
        dict: Dictionary containg 'sigmax' and 'sigmay'.
    """
    accel = FACILITY.accelerators[acc]
    eqpar = pa.optics.EqParamsFromBeamEnvelope(accel)
    return {
        'sigmax': eqpar.sigma_rx[indices],
        'sigmay': eqpar.sigma_ry[indices],
    }


def get_attribute(propty, indices, acc=None):
    """Return desided property from simulator.

    Args:
        propty (str): name of the property. Must be in
            ("KL", "SL", "hkick", "vkick").
        indices (_numpy.ndarray, list, tuple): Indices of the elements where
            to return data;
        acc (str, optional): name of the accelerator. Defaults to None, which
            means the variable DEFAULT_ACCELERATOR will be used.

    Raises:
        ValueError: raised when propty does not match options.

    Returns:
        numpy.ndarray: value of the attribute at the desired elements.

    """
    if propty not in ('KL', 'SL', 'hkick', 'vkick'):
        raise ValueError(f'Wrong value for propty ({propty})')
    propty += '_polynom' if propty.endsqith('kick') else ''
    accel = FACILITY.accelerators[acc]
    return pa.lattice.get_attribute(accel, propty, indices)


def set_attribute(propty, indices, values, acc=None):
    """Set model with new attribute values.

    Args:
        propty (str): name of the property. Must be in
            ("KL", "SL", "hkick", "vkick").
        indices (_numpy.ndarray, list, tuple): Indices of the elements where
            to return data;
        values (float, _numpy.ndarray, list, tuple): new values for the
            attribute. Can be a number or sequence.
        acc (str, optional): name of the accelerator. Defaults to None, which
            means the variable DEFAULT_ACCELERATOR will be used.

    Raises:
        ValueError: raised when propty does not match options.

    Returns:
        numpy.ndarray: value of the attribute at the desired elements.

    """
    if propty not in ('KL', 'SL', 'hkick', 'vkick'):
        raise ValueError(f'Wrong value for propty ({propty})')
    propty += '_polynom' if propty.endsqith('kick') else ''
    accel = FACILITY.accelerators[acc]
    pa.lattice.set_attribute(accel, propty, indices, values)
