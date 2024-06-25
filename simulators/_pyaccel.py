import numpy as _np
import pyaccel as pa

from .. import MODEL


def get_positions(indices):
    """Return the longitudinal position along the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        numpy.ndarray: position in [m] along the accelerator.
    """
    return _np.array(pa.lattice.find_spos(MODEL, indices=indices))


def get_orbit(indices):
    """Return the orbit of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        tuple: (orbx, orby) in [m].
    """
    orb = pa.tracking.find_orbit6(MODEL, indices=indices)
    return orb[0], orb[2]


def get_twiss(indices):
    """Return twiss parameters of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        dict: Dictionary containing Twiss parameters. Available keys:
            'betax', 'betay', 'alphax', 'alphay', 'mux', 'muy',
            'etax', 'etay', 'etapx', 'etapy'.
    """
    twi, *_ = pa.optics.calc_twiss(MODEL, indices=indices)
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


def get_beamsizes(indices):
    """Return beam sizes of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        dict: Dictionary containg 'sigmax' and 'sigmay'.
    """
    eqpar = pa.optics.EqParamsFromBeamEnvelope(MODEL)
    return {
        'sigmax': eqpar.sigma_rx[indices],
        'sigmay': eqpar.sigma_ry[indices],
    }
