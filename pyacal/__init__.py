"""."""

import importlib as _importlib

from ._control_systems import ControlSystemOptions
from ._facilities import FacilityOptions
from ._simulators import SimulatorOptions

FACILITY = None
SIMULATOR = None
CONTROL_SYSTEM = None


def set_facility(fac_name="sirius"):
    """Chose which accelerator will be abstracted.

    Args:
        fac_name (str, optional): Name of the accelerator to be abstracted.
            Possible options are:
                'sirius', 'esrf', 'soleil'

    """
    if fac_name.lower() not in FacilityOptions:
        raise ValueError(f"Wrong value for fac_name ({fac_name}).")
    global FACILITY
    fac_module = _importlib.import_module('._facilities.' + fac_name, __name__)
    FACILITY = fac_module.Facility
    _set_simulator(FACILITY.simulator)
    _set_control_system(FACILITY.control_system)


def get_model(acc):
    """Return the model of an accelerator.

    Returns:
        (paccel.accelerator.Accelerator | pyat.Accelerator): Model of the
            accelerator being abstracted.
    """
    return FACILITY.accelerators[acc]["model"]


def switch2online():
    """Switch ACAL to online mode."""
    _set_control_system(FACILITY.control_system)


def switch2simulation():
    """Switch ACAL to simulation mode."""
    _set_control_system("simulation")


def is_online():
    """Return True if in online mode.

    Returns:
        bool: whether ACAL is in online or simulation mode.
    """
    return CONTROL_SYSTEM.Name != "simulation"


def get_alias_map():
    """."""
    return FACILITY.alias_map


def get_alias_from_key(key, value, accelerator=None):
    """."""
    _check_key(key)
    acc = accelerator or FACILITY.default_accelerator
    return [
        alias
        for alias, amap in FACILITY.alias_map.items()
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
    _check_key(key)
    acc = accelerator or FACILITY.default_accelerator
    indices = []
    for _, amap in FACILITY.alias_map.items():
        if value in amap.get(key, []) and acc == amap.get("accelerator"):
            indices.append(amap["sim_info"]["indices"])
    return indices


def get_indices_from_alias(alias):
    """."""
    return [idx for idx in FACILITY.alias_map[alias]["sim_info"]["indices"]]


def get_alias_from_indices(indices, accelerator=None):
    """."""
    acc = accelerator or FACILITY.default_accelerator
    return [
        alias
        for alias, amap in FACILITY.alias_map.items()
        if indices in amap["sim_info"]["indices"]
        and acc == amap.get("accelerator")
    ]


def _check_key(key):
    if not any(key in amap for amap in FACILITY.alias_map.values()):
        raise ValueError(f"Key '{key}' not found in any alias_map entry.")


# ---------------------  private helper methods ----------------------------
def _set_simulator(simulator):
    """."""
    global SIMULATOR
    if simulator.lower() not in SimulatorOptions:
        raise ValueError(f'Wrong value for simulator ({simulator}).')
    SIMULATOR = _importlib.import_module('._simulators.' + simulator, __name__)


def _set_control_system(control_system):
    """."""
    global CONTROL_SYSTEM
    if control_system.lower() not in ControlSystemOptions:
        raise ValueError(f"Wrong value for fac_name ({control_system}).")

    if CONTROL_SYSTEM is None:
        CONTROL_SYSTEM = _importlib.import_module(
            '._control_systems.' + control_system, __name__)
        return
    elif control_system == CONTROL_SYSTEM.Name:
        return
    cs_new = _importlib.import_module(
        '._control_systems.' + control_system, __name__)

    for key in CONTROL_SYSTEM.ALL_CONNECTIONS:
        cs_new.PV(*key)
    CONTROL_SYSTEM = cs_new
