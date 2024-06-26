"""."""
import importlib as _importlib

from ._facilities import FacilityOptions
from ._simulators import SimulatorOptions
from ._control_system import ControlSystemOptions

FACILITY = None
SIMULATOR = None
CONTROL_SYSTEM = None


def set_facility(fac_name='sirius'):
    """Chose which accelerator will be abstracted.

    Args:
        fac_name (str, optional): Name of the accelerator to be abstracted.
            Possible options are:
                'sirius', 'esrf', 'soleil'

    """
    if fac_name.lower() not in FacilityOptions:
        raise ValueError(f'Wrong value for fac_name ({fac_name}).')
    global FACILITY
    fac_module = _importlib.import_module('._facilities', fac_name)
    FACILITY = fac_module.Facility
    _set_simulator(FACILITY.simulator)
    _set_control_system(FACILITY.control_system)


def get_model(acc):
    """Return the model of an accelerator.

    Returns:
        (paccel.accelerator.Accelerator | pyat.Accelerator): Model of the
            accelerator being abstracted.
    """
    return FACILITY.Accelerators[acc]['model']


def switch2online():
    """Switch ACAL to online mode."""
    _set_control_system(FACILITY.control_system)


def switch2simulation():
    """Switch ACAL to simulation mode."""
    _set_control_system('simulation')


def is_online():
    """Return True if in online mode.

    Returns:
        bool: whether ACAL is in online or simulation mode.
    """
    return CONTROL_SYSTEM.Name != 'simulation'


# ---------------------------- helper methods ------------------------------
def _set_simulator(simulator):
    """."""
    global SIMULATOR
    if simulator.lower() not in SimulatorOptions:
        raise ValueError(f'Wrong value for simulator ({simulator}).')
    SIMULATOR = _importlib.import_module('._simulator', simulator)


def _set_control_system(control_system):
    """."""
    global CONTROL_SYSTEM
    if control_system.lower() not in ControlSystemOptions:
        raise ValueError(f'Wrong value for fac_name ({control_system}).')

    if CONTROL_SYSTEM is None:
        CONTROL_SYSTEM = _importlib.import_module(
            '._control_system', control_system)
        return
    elif control_system == CONTROL_SYSTEM.Name:
        return
    cs_new = _importlib.import_module('._control_system', control_system)

    for key in CONTROL_SYSTEM.ALL_CONNECTIONS:
        cs_new.PV(*key)
    CONTROL_SYSTEM = cs_new
