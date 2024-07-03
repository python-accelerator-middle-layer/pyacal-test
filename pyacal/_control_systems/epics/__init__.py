"""Epics-related subpackage."""

from ... import _get_connections_dict
from .multiprocessing import ProcessSpawn
from .pv import PV as _PV
from .threading import Thread

del pv
del multiprocessing
del threading

Name = 'epics'


def create_pv(devname, propty, auto_monitor=True, connection_timeout=None):
    """Create connection with PV.

    Args:
        devname (str): Alias for device name defined in Facility.alias_map.
        propty (str): alias for property name defined in Facility.alias_map.
        auto_monitor (bool, optional): Whether to turn on auto monitor.
            Defaults to True.
        connection_timeout (int, optional): time to wait for connection before
            raising Timeout error. Defaults to None.

    Returns:
        tuple: key to access PV object with `get_pv` method.

    """
    args = (
        devname,
        propty,
        auto_monitor,
        connection_timeout
    )
    conns = _get_connections_dict()
    pvo = conns.get(args)
    if pvo is None:
        pvo = _PV(*args)
        conns[args] = pvo
    return args


def get_pv(key):
    """Retrieve PV from active connections.

    Args:
        key (tuple): Key to access object inside connections dictionary.

    Raises:
        KeyError: When PV is not defined yet.

    Returns:
        pvo (control_system.PV): PV object.
    """
    pvo = _get_connections_dict().get(key)
    if pvo is None:
        raise KeyError('PV not defined. Create it first with `create_pv`.')
    return pvo
