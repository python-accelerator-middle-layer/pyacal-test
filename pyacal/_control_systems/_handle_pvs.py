"""."""

from .. import _get_connections_dict, _get_control_system


def create_pv(devname, propty, connection_timeout=None):
    """Create connection with PV.

    Args:
        devname (str): Alias for device name defined in Facility.alias_map.
        propty (str): alias for property name defined in Facility.alias_map.
        connection_timeout (int, optional): time to wait for connection before
            raising Timeout error. Defaults to None.

    Returns:
        key (tuple, ): key to access PV object with `get_pv` method.

    """
    args = (
        devname,
        propty,
        connection_timeout
    )
    conns = _get_connections_dict()
    i = 0
    while True:
        key = (args, i)
        if not conns.get(key):
            conns[key] = _get_control_system().PV(*args)
            return key
        i += 1


def get_pv(key):
    """Retrieve PV from active connections.

    Args:
        key (tuple, ): Key to access object inside connections dictionary.

    Raises:
        KeyError: When PV is not defined yet.

    Returns:
        pvo (control_system.PV): PV object.
    """
    pvo = _get_connections_dict().get(key)
    if pvo is None:
        raise KeyError('PV not defined. Create it first with `create_pv`.')
    return pvo
