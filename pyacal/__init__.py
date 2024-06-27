"""."""

import importlib as _importlib

from ._control_systems import ControlSystemOptions
from ._facilities import FacilityOptions
from ._simulators import SimulatorOptions

# NOTE: Package-wide variables must be mutable objects in order to not be
# copied in import time. Thanks to steveha (accepted answer) answer in:
# https://stackoverflow.com/questions/1977362/
#           how-to-create-module-wide-variables-in-python
__ACAL_VARS = {
    'facility': None,
    'simulator': None,
    'control_system': None,
    'all_connections': dict(),
}


def set_facility(fac_name="sirius"):
    """Chose which accelerator will be abstracted.

    Args:
        fac_name (str, optional): Name of the accelerator to be abstracted.
            Possible options are:
                'sirius', 'esrf', 'soleil'

    """
    if fac_name.lower() not in FacilityOptions:
        raise ValueError(f"Wrong value for fac_name ({fac_name}).")
    fac_module = _importlib.import_module('._facilities.' + fac_name, __name__)
    __ACAL_VARS['facility'] = fac_module.facility
    __set_simulator(__ACAL_VARS['facility'].simulator)
    __set_control_system(__ACAL_VARS['facility'].control_system)


def get_model(acc):
    """Return the model of an accelerator.

    Returns:
        (paccel.accelerator.Accelerator | pyat.Accelerator): Model of the
            accelerator being abstracted.
    """
    cst = _get_control_system()
    return cst.accelerators[acc]


def switch2online():
    """Switch ACAL to online mode."""
    facil = _get_facility()
    __set_control_system(facil.control_system)


def switch2simulation():
    """Switch ACAL to simulation mode."""
    __set_control_system("simulation")


def is_online():
    """Return True if in online mode.

    Returns:
        bool: whether ACAL is in online or simulation mode.
    """
    cst = _get_control_system()
    return cst.Name != "simulation"


def get_alias_map():
    """."""
    facil = _get_facility()
    return facil.alias_map


def get_alias_from_key(key, value, accelerator=None):
    """."""
    __check_key(key)
    facil = _get_facility()
    acc = accelerator or facil.default_accelerator
    return [
        alias
        for alias, amap in facil.alias_map.items()
        if value in amap.get(key, []) and acc == amap.get("accelerator")
    ]


def get_alias_from_devname(devname, accelerator=None):
    """."""
    return get_alias_from_key("cs_devname", devname, accelerator=accelerator)


def get_alias_from_devtype(devtype, accelerator=None):
    """."""
    return get_alias_from_key("cs_devtype", devtype, accelerator=accelerator)


def get_alias_from_property(propty, accelerator=None):
    """."""
    return get_alias_from_key("cs_propties", propty, accelerator=accelerator)


def get_indices_from_key(key, value, accelerator=None):
    """."""
    __check_key(key)
    facil = _get_facility()
    acc = accelerator or facil.default_accelerator
    indices = []
    for _, amap in facil.alias_map.items():
        if value in amap.get(key, []) and acc == amap.get("accelerator"):
            indices.append(amap["sim_info"]["indices"])
    return indices


def get_indices_from_alias(alias):
    """."""
    facil = _get_facility()
    return [idx for idx in facil.alias_map[alias]["sim_info"]["indices"]]


def get_alias_from_indices(indices, accelerator=None):
    """."""
    facil = _get_facility()
    acc = accelerator or facil.default_accelerator
    return [
        alias
        for alias, amap in facil.alias_map.items()
        if indices in amap["sim_info"]["indices"]
        and acc == amap.get("accelerator")
    ]


# -------- functions used by rest of the package (not for the user) -----------
def _get_facility():
    """Return the current facility object being used."""
    facil = __ACAL_VARS['facility']
    if facil is None:
        raise RuntimeError(
            'Facility is not defined yet. Call `set_facility` function.'
        )
    return facil


def _get_simulator():
    """Return the current simulator object being used."""
    simul = __ACAL_VARS['simulator']
    if simul is None:
        raise RuntimeError(
            'Simulator is not defined yet. Call `set_facility` function.'
        )
    return simul


def _get_control_system():
    """Return the current control system being used."""
    cst = __ACAL_VARS['control_system']
    if cst is None:
        raise RuntimeError(
            'Control sytem is not defined yet. Call `set_facility` function.'
        )
    return cst


def _get_connections_dict():
    """Return the dictionary with all connections being used."""
    return __ACAL_VARS['all_connections']


# ---------------------  private helper methods ----------------------------
def __check_key(key):
    facil = _get_facility()
    if not any(key in amap for amap in facil.alias_map.values()):
        raise ValueError(f"Key '{key}' not found in any alias_map entry.")


def __set_simulator(simulator):
    """."""
    if simulator.lower() not in SimulatorOptions:
        raise ValueError(f'Wrong value for simulator ({simulator}).')
    __ACAL_VARS['simulator'] = _importlib.import_module(
        '._simulators.' + simulator, __name__
    )


def __set_control_system(control_system):
    """."""
    if control_system.lower() not in ControlSystemOptions:
        raise ValueError(f"Wrong value for fac_name ({control_system}).")

    if __ACAL_VARS['control_system'] is None:
        __ACAL_VARS['control_system'] = _importlib.import_module(
            '._control_systems.' + control_system, __name__)
        return
    elif control_system == __ACAL_VARS['control_system'].Name:
        return
    cs_new = _importlib.import_module(
        '._control_systems.' + control_system, __name__)

    all_conn = __ACAL_VARS['all_connections']
    __ACAL_VARS['all_connections'] = dict()
    for key in all_conn:
        cs_new.PV(*key)
    __ACAL_VARS['control_system'] = cs_new
