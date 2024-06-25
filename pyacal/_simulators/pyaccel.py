import numpy as _np
import pyaccel as pa

from .. import ACCELERATORS


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
    return _np.array(pa.lattice.find_spos(ACCELERATORS[acc], indices=indices))


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
    orb = pa.tracking.find_orbit6(ACCELERATORS[acc], indices=indices)
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
    twi, *_ = pa.optics.calc_twiss(ACCELERATORS[acc], indices=indices)
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
    eqpar = pa.optics.EqParamsFromBeamEnvelope(ACCELERATORS[acc])
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
    return pa.lattice.get_attribute(ACCELERATORS[acc], propty, indices)


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
    pa.lattice.set_attribute(ACCELERATORS[acc], propty, indices, values)
