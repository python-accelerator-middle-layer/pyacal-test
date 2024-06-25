from copy import deepcopy as _dcopy

import pymodels

from ..control import ControlSystemsOptions as _CSOptions
from ..simulators import SimulatorsOptions as _SimulOptions


def define_si():
    model = pymodels.si.create_accelerator()
    _famdata = pymodels.si.get_family_data(model)

    famdata = dict()
    famdata['BPM'] = dict()
    famdata['BPM']['indices'] = _famdata['BPM']['index']
    famdata['BPM']['devnames'] = _famdata['BPM']['devnames']
    famdata['BPM']['properties'] = dict(posx=':PosX-Mon', posy=':PosY-Mon')

    famdata['HCM'] = dict()
    famdata['HCM']['indices'] = _famdata['CH']['index']
    famdata['HCM']['devnames'] = _famdata['CH']['devnames']
    famdata['HCM']['properties'] = dict(
        current_sp=':Current-SP', current_rb=':CurrentRef-Mon')

    famdata['VCM'] = dict()
    famdata['VCM']['indices'] = _famdata['CV']['index']
    famdata['VCM']['devnames'] = _famdata['CV']['devnames']
    famdata['VCM']['properties'] = dict(
        current_sp=':Current-SP', current_rb=':CurrentRef-Mon')

    famdata['QN'] = dict()
    famdata['QN']['indices'] = _famdata['QN']['index']
    famdata['QN']['devnames'] = _famdata['QN']['devnames']
    famdata['QN']['properties'] = dict(
        current_sp=':Current-SP', current_rb=':CurrentRef-Mon')

    famdata['QS'] = dict()
    famdata['QS']['indices'] = _famdata['QS']['index']
    famdata['QS']['devnames'] = _famdata['QS']['devnames']
    famdata['QS']['properties'] = dict(
        current_sp=':Current-SP', current_rb=':CurrentRef-Mon')

    famdata['DCCT'] = dict()
    famdata['DCCT']['indices'] = _famdata['DCCT']['index'][:1]
    famdata['DCCT']['devnames'] = _famdata['DCCT']['devnames'][:1]
    famdata['DCCT']['properties'] = dict(current=':Current-Mon')

    alias_map = dict()

    for i, idcs in _famdata['BPM']['index']:
        devname = _famdata['BPM']['devnames'][i]
        alias = devname.dev + devname.get_nickname()
        alias_map[alias] = {
            'cs_devname': devname,
            'cs_devtype': 'BPM',
            'sim_info': {
                'accelerator': 'SI',
                'indices': idcs,
            },
            'cs_propties': {
                'posx': {
                    'name': ':PosX-Mon',
                    'conv_cs2sim': 1e-9,  # from [nm] to [m]
                }
            },
        }

    props = {
        'pwrstate_sp': {
            'name': ':PwrState-Sel',
            'conv_cs2si': None,
        },
        'pwrstate_rb': {
            'name': ':PwrState-Sts',
            'conv_cs2si': None,
        },
        'current_sp': {
            'name': ':Current-SP',
            'conv_cs2si': 1.0,
        },
        'current_rb': {
            'name': ':CurrentRef-Mon',
            'conv_cs2si': 1.0,
        },
        'current_mon': {
            'name': ':Current-Mon',
            'conv_cs2si': 1.0,
        },
    }
    for i, idcs in _famdata['QN']['index']:
        devname = _famdata['QN']['devnames'][i]
        alias = devname.dev + devname.get_nickname()
        alias_map[alias] = {
            'cs_devname': devname,
            'cs_devtype': 'Quadrupole Normal',
            'sim_info': {
                'accelerator': 'SI',
                'indices': idcs,
            },
            'cs_propties': _dcopy(props),
        }


    return model


def define_bo():
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)

    return model


def define_tb():
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)

    return model


def define_ts():
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)

    return model


def define_li():
    model = pymodels.tb.create_accelerator()
    famdata = pymodels.tb.get_family_data(model)

    return model


CONTROL_SYSTEM = _CSOptions.Epics
SIMULATOR = _SimulOptions.Pyaccel
ALIAS_MAP = {}
ACCELERATORS = {
    'SI': define_si(ALIAS_MAP),
    'BO': define_bo(ALIAS_MAP),
    'TB': define_tb(ALIAS_MAP),
    'TS': define_ts(ALIAS_MAP),
    'LI': define_li(ALIAS_MAP),
}
