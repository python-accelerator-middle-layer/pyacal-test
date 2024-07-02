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

    def get_alias_map(self):
        """."""
        return self.alias_map

    def check_alias_in_csdevtype(self, alias, cs_devtype):
        """Check if a given alias is part of a given cs_devtype."""
        return cs_devtype in self.alias_map[alias]["cs_devtype"]

    def get_alias_from_key(self, key, value, accelerator=None):
        """."""
        self.__check_key(key)
        acc = accelerator or self.default_accelerator
        return [
            alias
            for alias, amap in self.alias_map.items()
            if value in amap.get(key, []) and acc == amap.get("accelerator")
        ]

    def get_alias_from_devname(self, devname, accelerator=None):
        """."""
        return self.get_alias_from_key(
            "cs_devname", devname, accelerator=accelerator
        )

    def get_alias_from_devtype(self, devtype, accelerator=None):
        """."""
        return self.get_alias_from_key(
            "cs_devtype", devtype, accelerator=accelerator
        )

    def get_alias_from_property(self, propty, accelerator=None):
        """."""
        return self.get_alias_from_key(
            "cs_propties", propty, accelerator=accelerator
        )

    def get_indices_from_key(self, key, value, accelerator=None):
        """."""
        self.__check_key(key)
        acc = accelerator or self.default_accelerator
        indices = []
        for _, amap in self.alias_map.items():
            if value in amap.get(key, []) and acc == amap.get("accelerator"):
                indices.append(amap["sim_info"]["indices"])
        return indices

    def get_indices_from_alias(self, alias):
        """."""
        return [idx for idx in self.alias_map[alias]["sim_info"]["indices"]]

    def get_alias_from_indices(self, indices, accelerator=None):
        """."""
        acc = accelerator or self.default_accelerator
        return [
            alias
            for alias, amap in self.alias_map.items()
            if indices in amap["sim_info"]["indices"]
            and acc == amap.get("accelerator")
        ]

    def __check_key(self, key):
        if not any(key in Facility._AMAP_KEYS):
            raise ValueError(f"Key '{key}' not found in any alias_map entry.")
