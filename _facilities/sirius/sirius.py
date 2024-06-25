from copy import deepcopy as _dcopy

import pymodels

from .. import FacilityBase


def define_si(alias_map):
    """."""
    model = pymodels.si.create_accelerator()
    famdata = pymodels.si.get_family_data(model)

    devname = famdata['DCCT']['devnames'][0]
    alias = devname.dev + devname.get_nickname()
    if alias in alias_map:
        raise KeyError(f'Alias {alias} already defined.')
    alias_map[alias] = {
        'cs_devname': devname,
        'cs_devtype': 'DCCT',
        'sim_info': {
            'accelerator': 'SI',
            'indices': famdata['DCCT']['index'][0],
        },
        'cs_propties': {
            'posx': {
                'name': ':Current-Mon',
                'conv_cs2sim': 1e-9,  # from [mA] to [A]
            }
        },
    }

    for i, idcs in famdata['BPM']['index']:
        devname = famdata['BPM']['devnames'][i]
        alias = devname.dev + devname.get_nickname()
        if alias in alias_map:
            raise KeyError(f'Alias {alias} already defined.')
        alias_map[alias] = {
            'cs_devname': devname,
            'cs_devtype': 'BPM',
            'accelerator': 'SI',
            'sim_info': {
                'indices': idcs,
            },
            'cs_propties': {
                'posx': {
                    'name': ':PosX-Mon',
                    'conv_cs2sim': 1e-9,  # from [nm] to [m]
                },
                'posy': {
                    'name': ':PosY-Mon',
                    'conv_cs2sim': 1e-9,  # from [nm] to [m]
                },
            },
        }

    props = {
        'pwrstate_sp': {'name': ':PwrState-Sel', 'conv_cs2si': None},
        'pwrstate_rb': {'name': ':PwrState-Sts', 'conv_cs2si': None},
        'current_sp': {'name': ':Current-SP', 'conv_cs2si': 1.0},
        'current_rb': {'name': ':CurrentRef-Mon', 'conv_cs2si': 1.0},
        'current_mon': {'name': ':Current-Mon', 'conv_cs2si': 1.0},
    }
    typs = ['QN', 'QS', 'CH', 'CV']
    typ_names = [
        'Quadrupole Normal',
        'Quadrupole Skew',
        'Corrector Horizontal',
        'Corrector Vertical',
    ]
    for typ, name in zip(typs, typ_names):
        for i, idcs in famdata[typ]['index']:
            devname = famdata[typ]['devnames'][i]
            alias = devname.dev + devname.get_nickname()
            if alias in alias_map:
                raise KeyError(f'Alias {alias} already defined.')
            alias_map[alias] = {
                'cs_devname': devname,
                'cs_devtype': name,
                'accelerator': 'SI',
                'sim_info': {
                    'indices': idcs,
                },
                'cs_propties': _dcopy(props),
            }

    return model


def define_bo(alias_map):
    """."""
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)
    return model


def define_tb(alias_map):
    """."""
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)
    return model


def define_ts(alias_map):
    """."""
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)
    return model


def define_li(alias_map):
    """."""
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)
    return model


Facility = FacilityBase('sirius', 'epics', 'pyaccel')
Facility.default_accelerator = 'SI'
Facility.accelerators = {
    'SI': define_si(Facility.alias_map),
    'BO': define_bo(Facility.alias_map),
    'TB': define_tb(Facility.alias_map),
    'TS': define_ts(Facility.alias_map),
    'LI': define_li(Facility.alias_map),
}
