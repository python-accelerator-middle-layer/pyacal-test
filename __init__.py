"""."""

import _facilities

CONTROL_SYSTEM = 'epics'
SIMULATOR = 'pyaccel'
ACCELERATORS = dict()
ISONLINE = True
ALIAS_MAP = dict()


def set_facility(fac_name='SIRIUS'):
    """Chose which accelerator will be abstracted.

    Args:
        acc_name (str, optional): Name of the accelerator to be abstracted.
            Defaults to 'SIRIUS_SR'. Possible options are:
                'SIRIUS_SR', 'SIRIUS_BO', 'ESRF_SR', 'SOLEIL_SR'
    """
    if fac_name == "SIRIUS":
        fac = _facilities.sirius
    global CONTROL_SYSTEM, SIMULATOR, ACCELERATORS, ALIAS_MAP
    CONTROL_SYSTEM = fac.CONTROL_SYSTEM
    SIMULATOR = fac.SIMULATOR
    ACCELERATORS = fac.ACCELERATORS
    ALIAS_MAP = fac.ALIAS_MAP


def get_model(acc):
    """Return the model of an accelerator.

    Returns:
        (paccel.accelerator.Accelerator | pyat.Accelerator): Model of the
            accelerator being abstracted.
    """
    return ACCELERATORS[acc]['model']


def switch2online():
    """Switch ACAL to online mode."""
    global ISONLINE
    ISONLINE = True


def switch2offline():
    """Switch ACAL to offline mode."""
    global ISONLINE
    ISONLINE = False


def is_online():
    """Return True if in online mode.

    Returns:
        bool: whether on ACAL is in online or offline model.
    """
    return ISONLINE
