import at
import numpy

from .. import _get_facility


def get_positions(indices, acc=None):
    """Return the longitudinal position along the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        numpy.ndarray: position in [m] along the accelerator.
    """
    accel = _get_facility().accelerators[acc]
    return accel.get_s_pos(indices)


def get_orbit(indices, acc=None):
    """Return the orbit of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        tuple: (orbx, orby) in [m].
    """
    accel = _get_facility().accelerators[acc]
    _, o6 = accel.find_orbit(indices)
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
    accel = _get_facility().accelerators[acc]
    _, _, twi = accel.get_optics(indices)
    return {
        'betax': twi.beta[0, :],
        'betay': twi.beta[1, :],
        'alphax': twi.alpha[0, :],
        'alphay': twi.alpha[1, :],
        'mux': twi.mu[0, :]/(2*numpy.pi),
        'muy': twi.mu[1, :]/(2*numpy.pi),
        'etax': twi.dispersion[0, :],
        'etay': twi.dispersion[1, :],
        'etapx': twi.dispersion[2, :],
        'etapy': twi.dispersion[3, :],
    }


def get_beamsizes(indices, acc=None):
    """Return beam sizes of the model.

    Args:
        indices (_numpy.ndarray, list, tuple): Indices of the element in the
            model where to return data.

    Returns:
        dict: Dictionary containg 'sigmax' and 'sigmay'.
    """
    accel = _get_facility().accelerators[acc]
    _, _, twi = accel.get_optics(indices)
    bd = at.envelope_parameters(ring.enable_6d(copy=True))
    return {
        'sigmax': numpy.sqrt(twi.beta[0, :]*bd.emittances[0] +
                             (twi.dispersion[0, :]*bd.sigma_e)**2),
        'sigmay': numpy.sqrt(twi.beta[1, :]*bd.emittances[1]),
    }


def get_attribute(propty, indices, pol_order=None, acc=None):
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
    accel = _get_facility().accelerators[acc]       
    if propty in ('KL', 'SL') and pol_order is None:
        raise ValueError(f'Set polynom order for propty ({propty})')
    
    # expensive check   
    for e in ring[indices]:
        if not isinstance(e, at.Multipole):  
            raise ValueError(f'Array contains elements that are not magnets')  
        if numpy.sum(e.KickAngle) != 0.0:
            raise ValueError(f'Please set all KickAngle attributes to zero in the model')
        
    length = ring.get_value_refpts(indices, 'Length')    
    if propty=='KL':
        strength = ring.get_value_refpts(indices, 'PolynomB', index=pol_order)
        return strength*length
    elif propty=='SL':
        strength = ring.get_value_refpts(indices, 'PolynomA', index=pol_order)
        return strength*length       
    elif propty=='hkick':
        strength = ring.get_value_refpts(indices, 'PolynomB', index=0)
        return strength*length 
    elif propty=='vkick':
        strength = ring.get_value_refpts(indices, 'PolynomA', index=0)
        return strength*length 
    else:
        raise ValueError(f'Wrong value for propty ({propty})')  
    


def set_attribute(propty, indices, values, pol_order=None, acc=None):
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
    accel = _get_facility().accelerators[acc]       
    if propty in ('KL', 'SL') and pol_order is None:
        raise ValueError(f'Set polynom order for propty ({propty})')
        
    # expensive check   
    for e in ring[indices]:
        if not isinstance(e, at.Multipole):  
            raise ValueError(f'Array contains elements that are not magnets')  
        if numpy.sum(e.KickAngle) != 0.0:
            raise ValueError(f'Please set all KickAngle attributes to zero in the model')
        
    length = ring.get_value_refpts(indices, 'Length')    
    if propty=='KL':
        ring.set_value_refpts(indices, 'PolynomB', values/length, index=pol_order)
    elif propty=='SL':
        ring.set_value_refpts(indices, 'PolynomA', values/length, index=pol_order)      
    elif propty=='hkick':
        ring.set_value_refpts(indices, 'PolynomB', values, index=0)
    elif propty=='vkick':
        ring.set_value_refpts(indices, 'PolynomA', values, index=0) 
    else:
        raise ValueError(f'Wrong value for propty ({propty})') 
