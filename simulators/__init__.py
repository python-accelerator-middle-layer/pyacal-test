"""."""

from .. import SIMULATOR
from ..utils import get_namedtuple as _get_namedtuple
from . import _pyaccel, _pyat

SimulatorsOptions = _get_namedtuple(
    "SimulatorsOptions", ("pyaccel", "pyat"))


def get_positions(indices):
    """Return the longitudinal position along the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        numpy.ndarray: position in [m] along the accelerator.
    """
    if SIMULATOR == SimulatorsOptions.pyaccel:
        return _pyaccel.get_positions(indices)
    elif SIMULATOR == SimulatorsOptions.pyat:
        return _pyat.get_positions(indices)
    else:
        raise ValueError("Chosen simulator is not installed.")


def get_orbit(indices):
    """Return the orbit of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        tuple: (orbx, orby) in [m].
    """
    if SIMULATOR == SimulatorsOptions.pyaccel:
        return _pyaccel.get_orbit(indices)
    elif SIMULATOR == SimulatorsOptions.pyat:
        return _pyat.get_orbit(indices)
    else:
        raise ValueError("Chosen simulator is not installed.")


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
    if SIMULATOR == SimulatorsOptions.pyaccel:
        return _pyaccel.get_twiss(indices)
    elif SIMULATOR == SimulatorsOptions.pyat:
        return _pyat.get_twiss(indices)
    else:
        raise ValueError("Chosen simulator is not installed.")


def get_beamsizes(indices):
    """Return beam sizes of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        dict: Dictionary containg 'sigmax' and 'sigmay'.
    """
    if SIMULATOR == SimulatorsOptions.pyaccel:
        return _pyaccel.get_beamsizes(indices)
    elif SIMULATOR == SimulatorsOptions.pyat:
        return _pyat.get_beamsizes(indices)
    else:
        raise ValueError("Chosen simulator is not installed.")
