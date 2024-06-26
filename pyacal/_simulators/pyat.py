import pyat as _pyat

from .. import FACILITY


def get_positions(indices):
    """Return the longitudinal position along the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        numpy.ndarray: position in [m] along the accelerator.
    """
    raise NotImplementedError("Please, implement me.")


def get_orbit(indices):
    """Return the orbit of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        tuple: (orbx, orby) in [m].
    """
    raise NotImplementedError("Please, implement me.")


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
    raise NotImplementedError("Please, implement me.")


def get_beamsizes(indices):
    """Return beam sizes of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        dict: Dictionary containg 'sigmax' and 'sigmay'.
    """
    raise NotImplementedError("Please, implement me.")


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
    raise NotImplementedError("Please, implement me.")


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
    raise NotImplementedError("Please, implement me.")
