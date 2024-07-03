"""Module to implement class Facility."""
from copy import deepcopy as _dcopy

from .. import _get_simulator
from ..utils import get_namedtuple as _get_namedtuple


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

    _AMAP_DEF = {
        'cs_devname': str,
        'cs_devtype': set,
        'accelerator': str,
        'sim_info': dict,
        'cs_propties': dict
    }

    def __init__(self, name, control_system, simulator):
        """."""
        self.name = name
        self.control_system = control_system
        self.simulator = simulator
        self._alias_map = {}
        self.accelerators = {}
        self.default_accelerator = ""

    def add_2_alias_map(self, key, value):
        """."""
        mapdef = self._AMAP_DEF
        if not isinstance(key, str):
            raise TypeError('Alias name should be of type str.')
        elif not isinstance(value, dict):
            raise TypeError('Alias map should be of type dict.')

        if key in self._alias_map:
            raise ValueError(f'Key {key} already in aliasmap.')

        if mapdef.keys() - value.keys():
            raise KeyError(f'Not all required keys are defined for {key}')

        for k, v in value.items():
            if k not in mapdef:
                raise KeyError(f'Key {k} present in {key} is not allowed.')
            elif not isinstance(v, mapdef[k]):
                raise ValueError(
                    f'Value of key {k} should be of type {str(mapdef[k])}.'
                )
        self._alias_map[key] = value

    def find_aliases_from_accelerator(self, accelerator, aliases=None):
        if aliases is None:
            aliases = list(self._alias_map)
        if isinstance(accelerator, str):
            accelerator = (accelerator, )
        res = []
        for alias in aliases:
            if self._alias_map[alias]['accelerator'] in accelerator:
                res.append(alias)
        return res

    def find_aliases_from_cs_devname(self, cs_devname, aliases=None):
        """."""
        if aliases is None:
            aliases = list(self._alias_map)
        if isinstance(cs_devname, str):
            cs_devname = (cs_devname, )
        res = []
        for alias in aliases:
            if self._alias_map[alias]['cs_devname'] in cs_devname:
                res.append(alias)
        return res

    def find_aliases_from_cs_devtype(
        self, cs_devtype, aliases=None, comp='and'
    ):
        if comp == 'or':
            def meth(x, y):
                return len(x | y) < len(x) + len(y)
        elif comp == 'and':
            def meth(x, y):
                return not bool(y - x)

        if aliases is None:
            aliases = list(self._alias_map)

        if isinstance(cs_devtype, str):
            cs_devtype = {cs_devtype, }
        elif isinstance(cs_devtype, (list, tuple)):
            cs_devtype = set(cs_devtype)

        res = []
        for alias in aliases:
            val_ref = self._alias_map[alias]['cs_devtype']
            if meth(val_ref, cs_devtype):
                res.append(alias)
        return res

    def get_attribute_from_aliases(self, attr, aliases=None):
        if aliases is None:
            aliases = list(self._alias_map)

        if isinstance(attr, str):
            attr = attr.split('.')

        res = []
        for alias in aliases:
            mapp = self._alias_map[alias]
            for val in attr:
                mapp = mapp[val]
            res.append(_dcopy(mapp))
        return res

    def is_alias_in_cs_devtype(self, alias, cs_devtype):
        return cs_devtype in self._alias_map[alias]['cs_devtype']

    def sort_aliases_by_model_positions(self, aliases):
        idcs = self.get_attribute_from_aliases(
            'sim_info.indices', aliases=aliases
        )
        simul = _get_simulator()
        pos = []
        for idx in idcs:
            pos.append(simul.get_positions(idx))
        aliases, _ = zip(*sorted(zip(aliases, pos), key=lambda x: x[1]))
        return aliases
