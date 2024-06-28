"""."""

from ..utils import get_namedtuple as _get_namedtuple

FacilityOptions = ("esrf", "sirius", "soleil")


class Facility:
    """."""

    _CS_DEVTYPES = (
        'PowerSupply',
        'DipoleNormal',
        'DipoleReverse',
        'DipoleSkew',
        'CorrectorHorizontal',
        'CorrectorVertical',
        'QuadrupoleNormal',
        'QuadrupoleSkew',
        'SextupoleNormal',
        'SextupoleSkew',
        'OctupoleNormal',
        'OctupoleSkew',
        'DCCT',
        'BPM',
        'IDBPM',
        'PBPM',
        'RFGenerator',
        'RFCavity',
        'TuneMeas',
        'SOFB',
    )
    CSDevTypes = _get_namedtuple('CSDevTypes', _CS_DEVTYPES, _CS_DEVTYPES)

    _AMAP_KEYS = {
        'cs_devname', 'cs_devtype', 'accelerator', 'sim_info', 'cs_propties'
    }

    def __init__(self, name, control_system, simulator):
        """."""
        self.name = name
        self.control_system = control_system
        self.simulator = simulator
        self.alias_map = dict()
        self.accelerators = dict()
        self.default_accelerator = ""

    def add2aliasmap(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Key should be of type str.')
        if key in self.alias_map:
            raise ValueError(f'Key {key} already in aliasmap.')

        if Facility._AMAP_KEYS - value.keys():
            raise KeyError(f'Not all required keys are defined for {key}')
        for k, _ in value.items():
            if k not in Facility._AMAP_KEYS:
                raise KeyError(f'Key {k} present in {key} is not allowed.')
        self.alias_map[key] = value
