"""Module to implement class Facility."""
from copy import deepcopy as _dcopy

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
        'Family'
    )
    CSDevTypes = _get_namedtuple('CSDevTypes', _CS_DEVTYPES, _CS_DEVTYPES)

    _AMAP_DEF = {
        'cs_devname': str,
        'cs_devtype': set,
        'accelerator': str,
        'sim_info': dict,
        'cs_propties': dict,
    }

    def __init__(self, name, control_system, simulator):
        """."""
        self.name = name
        self.control_system = control_system
        self.simulator = simulator
        self._alias_map = {}
        self.accelerators = {}
        self.default_accelerator = ""
        if control_system == 'tango':
            self._CONNECTED_DS = {}

    def add_2_alias_map(self, alias, value):
        """."""
        self._check_map_entry(alias, value)
        self._alias_map[alias] = value

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
        single = False
        if aliases is None:
            aliases = list(self._alias_map)
        if isinstance(aliases, str):
            single = True
            aliases = [aliases]

        if isinstance(attr, str):
            attr = attr.split('.')

        res = []
        for alias in aliases:
            mapp = self._alias_map[alias]
            for val in attr:
                mapp = mapp[val]
            res.append(_dcopy(mapp))
        if single:
            return res[0]
        return res

    def is_alias_in_cs_devtype(self, alias, cs_devtype):
        return cs_devtype in self._alias_map[alias]['cs_devtype']

    def sort_aliases_by_model_positions(self, aliases):
        from .. import _get_simulator

        idcs = self.get_attribute_from_aliases(
            'sim_info.indices', aliases=aliases
        )
        simul = _get_simulator()
        pos = []
        for idx, alias in zip(idcs, aliases):
            pos.append(
                simul.get_positions(
                    idx, acc=self._alias_map[alias]['accelerator']
                )
            )
        aliases, _ = zip(*sorted(zip(aliases, pos), key=lambda x: x[1]))
        return aliases

    # ---------------------- helper methods ---------------------------
    def _check_map_entry(self, alias, value):
        mapdef = self._AMAP_DEF
        if not isinstance(alias, str):
            raise TypeError('Alias name should be of type str.')
        elif not isinstance(value, dict):
            raise TypeError('Alias map should be of type dict.')

        if alias in self._alias_map:
            raise ValueError(f'Key {alias} already in aliasmap.')

        if mapdef.keys() - value.keys():
            raise KeyError(f'Not all required keys are defined for {alias}')

        for key, val in value.items():
            self._check_entry_keys(alias, key, val)

    def _check_entry_keys(self, alias, key, val):
        mapdef = self._AMAP_DEF
        if key not in mapdef:
            raise KeyError(f'Key {key} present in {alias} is not allowed.')
        try:
            val = mapdef.get(key)(val)
        except ValueError as e:
            e.args = (f'Value of key {key} '
                      f'should be of type {str(mapdef[key])}.', )
            raise

        if key == 'sim_info':
            self._check_sim_info(alias, val)
        elif key == 'cs_propties':
            for propty, val2 in val.items():
                self._check_cs_propties(alias, propty, val2)

    def _check_sim_info(self, alias, val):
        if not isinstance(val, dict):
            raise TypeError(f"Wrong type for `sim_info` of {alias}")

        all_keys = {'indices'}
        extra_keys = val.keys() - all_keys
        if extra_keys:
            raise KeyError(
                f"Indices of {alias} has extra keys. "
                f"Possible values are {all_keys}"
            )

        val = val.get('indices', [])
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"Wrong type for `indices` of {alias}")

    def _check_cs_propties(self, alias, propty, val):

        if not isinstance(val, dict):
            raise TypeError(
                f'value for propty {propty} of alias {alias} should be a dict.'
            )
        if 'name' not in val:
            raise KeyError(
                f'Propty {propty} of {alias} does not have `name` defined.'
            )
        if not isinstance(val['name'], str):
            raise TypeError(
                f'Name of propty {propty} of {alias} should be of type str.'
            )
        elif not isinstance(val.get('index', 0), int):
            raise TypeError(
                f'index of propty {propty} of {alias} should be of type int.'
            )
        elif not isinstance(val.get('wvalue', False), bool):
            raise TypeError(
                f'wvalue of propty {propty} of {alias} should be of type bool.'
            )

        all_keys = {'name', 'conv_sim2cs', 'conv_cs2phys', 'index', 'wvalue'}
        extra_keys = val.keys() - all_keys
        if extra_keys:
            raise KeyError(
                f"Propty {propty} of {alias} has extra keys. "
                f"Possible values are {all_keys}"
            )
