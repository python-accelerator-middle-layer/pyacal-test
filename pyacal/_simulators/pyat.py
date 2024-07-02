import at
import numpy

from .. import _get_facility


def get_positions(refpts, acc=None):
    """Return the longitudinal position along the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        numpy.ndarray: position in [m] along the accelerator.
    """
    facility = _get_facility()
    acc = acc or facility.default_accelerator
    accel = facility.accelerators[acc]
    return accel.get_s_pos(refpts)


def get_orbit(refpts, acc=None):
    """Return the orbit of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        tuple: (orbx, orby) in [m].
    """
    facility = _get_facility()
    acc = acc or facility.default_accelerator
    accel = facility.accelerators[acc]
    _, o6 = accel.find_orbit(refpts)
    return o6[:, 0], o6[:, 2]


def get_twiss(indices, acc=None):
    """Return twiss parameters of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        dict: Dictionary containing Twiss parameters. Available keys:
            'betax', 'betay', 'alphax', 'alphay', 'mux', 'muy',
            'etax', 'etay', 'etapx', 'etapy'.
    """
    facility = _get_facility()
    acc = acc or facility.default_accelerator
    accel = facility.accelerators[acc]
    _, _, twi = accel.get_optics(indices)
    return {
        'betax': twi.beta[:, 0],
        'betay': twi.beta[:, 1],
        'alphax': twi.alpha[:, 1],
        'alphay': twi.alpha[:, 1],
        'mux': twi.mu[:, 0]/(2*numpy.pi),
        'muy': twi.mu[:, 1]/(2*numpy.pi),
        'etax': twi.dispersion[:, 0],
        'etay': twi.dispersion[:, 1],
        'etapx': twi.dispersion[:, 2],
        'etapy': twi.dispersion[:, 3],
    }


def get_beamsizes(refpts, acc=None):
    """Return beam sizes of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        dict: Dictionary containg 'sigmax' and 'sigmay'.
    """
    facility = _get_facility()
    acc = acc or facility.default_accelerator
    accel = facility.accelerators[acc]
    _, _, twi = accel.get_optics(refpts)
    bd = at.envelope_parameters(accel.enable_6d(copy=True))
    return {
        'sigmax': numpy.sqrt(twi.beta[:, 0]*bd.emittances[0] +
                             (twi.dispersion[:, 1]*bd.sigma_e)**2),
        'sigmay': numpy.sqrt(twi.beta[:, 1]*bd.emittances[1]),
    }


def get_attribute(propty, refpts, index=None, acc=None):
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
    _ATTR = {'KL': ['PolynomB', index or None],
             'SL': ['PolynomA', index or None],
             'hkick': ['KickAngle', 0],
             'vkick': ['KickAngle', 1]}

    def get_attr(element, attr, attr_index):
        if attr_index is None and 'Polynom' in attr:
            attr_index = getattr(element, 'DefaultOrder', None)
            if attr_index is None:
                raise ValueError(f'Set polynom order for element ({element.FamName})')
        le = 1
        if 'Polynom' in attr: le = element.Length
        return getattr(element, attr)[attr_index] * le

    facility = _get_facility()
    acc = acc or facility.default_accelerator
    accel = facility.accelerators[acc]
    try:
        attr, attr_index = _ATTR[propty]
        return [get_attr(e, attr, attr_index) for e in accel[refpts]]
    except KeyError:
        raise ValueError(f'Wrong value for propty ({propty})')
    

def set_attribute(propty, refpts, values, index=None, acc=None):
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
    _ATTR = {'KL': ['PolynomB', index or None],
             'SL': ['PolynomA', index or None],
             'hkick': ['KickAngle', 0],
             'vkick': ['KickAngle', 1]}

    def set_attr(element, value, attr, attr_index):
        if attr_index is None and 'Polynom' in attr:
            attr_index = getattr(element, 'DefaultOrder', None)
            if attr_index is None:
                raise ValueError(f'Set polynom order for element ({element.FamName})')
        le = 1
        if 'Polynom' in attr: le = element.Length
        getattr(element, attr)[attr_index] = value / le

    facility = _get_facility()
    acc = acc or facility.default_accelerator
    accel = facility.accelerators[acc]
    values = numpy.broadcast_to(values, accel.refcount(refpts))
    try:
        attr, attr_index = _ATTR[propty]
        return [set_attr(e, v, attr, attr_index) for e, v in zip(accel[refpts], values)]
    except KeyError:
        raise ValueError(f'Wrong value for propty ({propty})')