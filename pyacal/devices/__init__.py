"""."""

from .bpm import BPM
from .dcct import DCCT
from .fambpms import FamBPMs
from .power_supply import PowerSupply
from .sofb import SOFB

_DEVICES = {
    'BPM': BPM,
    'DCCT': DCCT,
    'FamBPMs': FamBPMs,
    'PowerSupply': PowerSupply,
    'SOFB': SOFB,
}


def get_device_class(classname):
    """Get class for the Device type.

    Args:
        classname (str): Type of the device to return.

    Returns:
        class: Class of the device type.
    """
    return _DEVICES[classname]


def add_more_device_classes(class_dict):
    """Add external devices to the possible devices.

    Args:
        class_dict (dict): Dictionary whose keys are devices names
            and values are classes of Device or DeviceSet.
    """
    _DEVICES.update(class_dict)
