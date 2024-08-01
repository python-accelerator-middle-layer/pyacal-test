"""Define SIRIUS facility object with aliases map."""

from copy import deepcopy as _dcopy

import numpy as np
import pymodels
from scipy.constants import speed_of_light
from siriuspy.magnet.excdata import ExcitationData
from siriuspy.search import PSSearch

from ..._conversions.utils import ConverterNames
from ..facility import Facility

CSDevTypes = Facility.CSDevTypes


def define_si(facil: Facility):
    """."""
    model = pymodels.si.create_accelerator()
    model = pymodels.si.fitted_models.vertical_dispersion_and_coupling(model)
    model.cavity_on = True
    model.radiation_on = 1
    model.vchamber_on = True
    facil.accelerators['SI'] = model

    famdata = pymodels.si.get_family_data(model)

    devname = famdata['DCCT']['devnames'][0]
    alias = devname.dev + '-' + devname.get_nickname()
    facil.add_2_alias_map(
        alias,
        {
            'cs_devname': devname,
            'cs_devtype': {CSDevTypes.DCCT, },
            'accelerator': 'SI',
            'sim_info': {'indices': [famdata['DCCT']['index'][0]]},
            'cs_propties': {
                'current': {
                    'name': ':Current-Mon',
                    'conv_cs2sim': 1e-3,  # from [mA] to [A]
                }
            },
        }
    )

    # --------- Define BPMs ------------
    for i, idcs in enumerate(famdata['BPM']['index']):
        devname = famdata['BPM']['devnames'][i]
        alias = devname.dev + '-' + devname.get_nickname()
        facil.add_2_alias_map(
            alias,
            {
                'cs_devname': devname,
                'cs_devtype': {CSDevTypes.BPM, CSDevTypes.SOFB},
                'accelerator': 'SI',
                'sim_info': {'indices': [idcs]},
                'cs_propties': {
                    'posx': {
                        'name': ':PosX-Mon',
                        'conv_cs2sim': 1e-9,  # from [nm] to [m]
                        'conv_cs2phys': 1e-3,  # from [nm] to [um]
                    },
                    'posy': {
                        'name': ':PosY-Mon',
                        'conv_cs2sim': 1e-9,  # from [nm] to [m]
                        'conv_cs2phys': 1e-3,  # from [nm] to [um]
                    },
                },
            }
        )

    # --------- Define magnets (power supplies) ------------

    # ------------------- Bending Magnets -----------------------
    bc_intfield = 0.7503  # [T.m]
    bc_deflection = bc_intfield * speed_of_light / 3e9  # [rad]
    nominal_deflection = 2*np.pi / 20 - bc_deflection  # B1 + B2 deflection
    # Notes on the equation below:
    #  - factor of 2 needed because lookup table accounts for half integral;
    #  - the factor 1e9 needed to have energy in GeV instead of eV;
    scale_fac = 2 * speed_of_light / nominal_deflection / 1e9
    convs = [
        {
            'type': ConverterNames.LookupTableConverter,
            'table_name': 'name_of_excitation_table',
        },
        scale_fac,
    ]
    props = {
        'pwrstate_sp': {'name': ':PwrState-Sel'},
        'pwrstate_rb': {'name': ':PwrState-Sts'},
        'current_sp': {'name': ':Current-SP', 'conv_cs2sim': _dcopy(convs)},
        'current_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2sim': _dcopy(convs)},
        'current_mon': {'name': ':Current-Mon', 'conv_cs2sim': _dcopy(convs)},
        'energy_sp': {'name': ':Current-SP', 'conv_cs2phys': _dcopy(convs)},
        'energy_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2phys': _dcopy(convs),
        },
        'energy_mon': {
            'name': ':Current-Mon', 'conv_cs2phys': _dcopy(convs)
        },
    }
    typs = ['B1B2-1', 'B1B2-2']
    for typ in typs:
        devname = f'SI-Fam:PS-{typ}'
        alias = 'Fam-' + typ
        table_name = PSSearch.conv_psname_2_pstype(devname) + '.txt'
        prps = _dcopy(props)
        for prp in prps.values():
            con = prp.get('conv_cs2phys')
            if con is None:
                con = prp.get('conv_cs2sim')
            if con is None:
                continue
            con[0]['table_name'] = table_name
        facil.add_2_alias_map(
            alias,
            {
                'cs_devname': devname,
                'cs_devtype': {
                    CSDevTypes.DipoleNormal,
                    CSDevTypes.MagnetFamily,
                    CSDevTypes.PowerSupply,
                },
                'accelerator': 'SI',
                'sim_info': {'indices': famdata[typ]['index']},
                'cs_propties': prps,
            }
        )

    # ---------------- Quadrupole and Sextupole Families ------------------
    convs = [
        {
            'type': ConverterNames.LookupTableConverter,
            'table_name': 'name_of_excitation_table',
        },
        {
            'type': ConverterNames.MagRigidityConverter,
            'devname': 'Fam-B1B2-1',
            'propty': 'energy_rb',
            'conv_2_ev': 1e9,  # to convert from [GeV] to [eV]
        },
    ]
    props = {
        'pwrstate_sp': {'name': ':PwrState-Sel'},
        'pwrstate_rb': {'name': ':PwrState-Sts'},
        'current_sp': {'name': ':Current-SP', 'conv_cs2sim': _dcopy(convs)},
        'current_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2sim': _dcopy(convs)},
        'current_mon': {'name': ':Current-Mon', 'conv_cs2sim': _dcopy(convs)},
        'strength_sp': {'name': ':Current-SP', 'conv_cs2phys': _dcopy(convs)},
        'strength_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2phys': _dcopy(convs),
        },
        'strength_mon': {
            'name': ':Current-Mon', 'conv_cs2phys': _dcopy(convs)
        },
    }
    typs = [
        'Q1', 'Q2', 'Q3', 'Q4',
        'QFA', 'QDA',
        'QDB1', 'QFB', 'QDB2',
        'QDP1', 'QFP', 'QDP2',
        'SFA0', 'SDA0', 'SDA1', 'SFA1', 'SDA2', 'SDA3', 'SFA2',
        'SFB2', 'SDB3', 'SDB2', 'SFB1', 'SDB1', 'SDB0', 'SFB0',
        'SFP2', 'SDP3', 'SDP2', 'SFP1', 'SDP1', 'SDP0', 'SFP0',
    ]
    for typ in typs:
        devname = f'SI-Fam:PS-{typ}'
        alias = 'Fam-' + typ
        name = CSDevTypes.QuadrupoleNormal if typ.startswith('Q') \
            else CSDevTypes.SextupoleNormal
        table_name = PSSearch.conv_psname_2_pstype(devname) + '.txt'
        prps = _dcopy(props)
        for prp in prps.values():
            con = prp.get('conv_cs2phys')
            if con is None:
                con = prp.get('conv_cs2sim')
            if con is None:
                continue
            con[0]['table_name'] = table_name

        facil.add_2_alias_map(
            alias,
            {
                'cs_devname': devname,
                'cs_devtype': {
                    name,
                    CSDevTypes.MagnetFamily,
                    CSDevTypes.PowerSupply},
                'accelerator': 'SI',
                'sim_info': {'indices': famdata[typ]['index']},
                'cs_propties': prps,
            }
        )

    # ------------------ Slow Corrector Magnets ------------
    convs = [
        {
            'type': ConverterNames.LookupTableConverter,
            'table_name': 'name_of_excitation_table',
        },
        {
            'type': ConverterNames.MagRigidityConverter,
            'devname': 'Fam-B1B2-1',
            'propty': 'energy_rb',
            'conv_2_ev': 1e9,  # to convert from [GeV] to [eV]
        },
        1e6,  # from [rad] to [urad]
    ]
    cs2sim = _dcopy(convs)[:2]
    props = {
        'pwrstate_sp': {'name': ':PwrState-Sel'},
        'pwrstate_rb': {'name': ':PwrState-Sts'},
        'current_sp': {
            'name': ':Current-SP', 'conv_cs2sim': _dcopy(cs2sim),
        },
        'current_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2sim': _dcopy(cs2sim),
        },
        'current_mon': {
            'name': ':Current-Mon', 'conv_cs2sim': _dcopy(cs2sim)
        },
        'strength_sp': {
            'name': ':Current-SP', 'conv_cs2phys': _dcopy(convs)
        },
        'strength_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2phys': _dcopy(convs),
        },
        'strength_mon': {
            'name': ':Current-Mon', 'conv_cs2phys': _dcopy(convs)
        },
    }
    typs = ['CH', 'CV']
    typ_names = [
        CSDevTypes.CorrectorHorizontal,
        CSDevTypes.CorrectorVertical,
    ]
    for typ, name in zip(typs, typ_names):
        for i, idcs in enumerate(famdata[typ]['index']):
            devname = famdata[typ]['devnames'][i]
            alias = devname.dev + '-' + devname.get_nickname()

            table_name = PSSearch.conv_psname_2_pstype(devname) + '.txt'
            prps = _dcopy(props)
            for prp in prps.values():
                con = prp.get('conv_cs2phys')
                if con is None:
                    con = prp.get('conv_cs2sim')
                if con is None:
                    continue
                con[0]['table_name'] = table_name

            facil.add_2_alias_map(
                alias,
                {
                    'cs_devname': devname,
                    'cs_devtype': {
                        name,
                        CSDevTypes.MagnetIndividual,
                        CSDevTypes.PowerSupply,
                        CSDevTypes.SOFB,
                    },
                    'accelerator': 'SI',
                    'sim_info': {'indices': [idcs]},
                    'cs_propties': prps,
                }
            )

    # ----------------- Skew Quadrupoles --------------------------
    typ = 'QS'
    for i, idcs in enumerate(famdata[typ]['index']):
        devname = famdata[typ]['devnames'][i]
        alias = devname.dev + '-' + devname.get_nickname()
        table_name = PSSearch.conv_psname_2_pstype(devname) + '.txt'
        prps = _dcopy(props)
        for prp in prps.values():
            con = prp.get('conv_cs2phys')
            if con is None:
                con = prp.get('conv_cs2sim')
            if con is None:
                continue
            con[0]['table_name'] = table_name

        facil.add_2_alias_map(
            alias,
            {
                'cs_devname': devname,
                'cs_devtype': {
                    CSDevTypes.QuadrupoleSkew,
                    CSDevTypes.MagnetIndividual,
                    CSDevTypes.PowerSupply,
                },
                'accelerator': 'SI',
                'sim_info': {'indices': [idcs]},
                'cs_propties': _dcopy(prps),
            }
        )

    # --------------------- Normal Quadrupoles -------------------
    convs = [
        {
            'type': ConverterNames.LookupTableConverter,
            'table_name': 'name_of_excitation_table',
        },
        {
            'type': ConverterNames.MagRigidityConverter,
            'devname': 'Fam-B1B2-1',
            'propty': 'energy_rb',
            'conv_2_ev': 1e9,
        },
        {
            'type': ConverterNames.CompanionProptyConverter,
            'devname': 'Fam-QFA',
            'propty': 'strength_rb',
            'operation': 'add',
        },
    ]
    props = {
        'pwrstate_sp': {'name': ':PwrState-Sel'},
        'pwrstate_rb': {'name': ':PwrState-Sts'},
        'current_sp': {'name': ':Current-SP', 'conv_cs2sim': _dcopy(convs)},
        'current_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2sim': _dcopy(convs)
        },
        'current_mon': {'name': ':Current-Mon', 'conv_cs2sim': _dcopy(convs)},
        'strength_sp': {'name': ':Current-SP', 'conv_cs2phys': _dcopy(convs)},
        'strength_rb': {
            'name': ':CurrentRef-Mon', 'conv_cs2phys': _dcopy(convs)
        },
        'strength_mon': {
            'name': ':Current-Mon', 'conv_cs2phys': _dcopy(convs)
        },
    }

    typ = 'QN'
    for i, idcs in enumerate(famdata[typ]['index']):
        devname = famdata[typ]['devnames'][i]
        alias = devname.dev + '-' + devname.get_nickname()
        table_name = PSSearch.conv_psname_2_pstype(devname) + '.txt'
        mapp = {
            'cs_devname': devname,
            'cs_devtype': {
                CSDevTypes.QuadrupoleNormal,
                CSDevTypes.MagnetIndividual,
                CSDevTypes.PowerSupply,
            },
            'accelerator': 'SI',
            'sim_info': {'indices': [idcs]},
            'cs_propties': _dcopy(props),
        }
        for prp in mapp['cs_propties'].values():
            con = prp.get('conv_cs2phys')
            if con is None:
                con = prp.get('conv_cs2sim')
            if con is None:
                continue
            con[0]['table_name'] = table_name
            con[2]['devname'] = 'Fam-' + devname.dev

        facil.add_2_alias_map(alias, mapp)

    # -------- Define RF --------
    facil.add_2_alias_map(
        "RFGen",
        {
            'cs_devname': 'RF-Gen',
            'cs_devtype': {CSDevTypes.RFGenerator, },
            'accelerator': 'SI',
            'sim_info': {'indices': famdata['SRFCav']['index']},
            'cs_propties': {
                'frequency_rb': {'name': ':GeneralFreq-RB'},
                'frequency_sp': {'name': ':GeneralFreq-SP'},
            },
        }
    )

    phs_conv = 180 / np.pi
    facil.add_2_alias_map(
        "RFCav",
        {
            'cs_devname': 'SR-RF-DLLRF-01',
            'cs_devtype': {CSDevTypes.RFCavity, },
            'accelerator': 'SI',
            'sim_info': {'indices': famdata['SRFCav']['index']},
            'cs_propties': {
                'voltage_mon': {'name': ':SL:INP:AMP', 'conv_cs2sim': 1e-6},
                'voltage_rb': {'name': ':SL:REF:AMP', 'conv_cs2sim': 1e-6},
                'voltage_sp': {'name': ':mV:AL:REF-SP', 'conv_cs2sim': 1e-6},
                'phase_mon': {'name': ':SL:INP:PHS', 'conv_cs2sim': phs_conv},
                'phase_rb': {'name': ':SL:REF:PHS', 'conv_cs2sim': phs_conv},
                'phase_sp': {'name': ':PL:REF:S', 'conv_cs2sim': phs_conv},
            },
        }
    )

    # -------- Define Tune Measurement Device --------
    facil.add_2_alias_map(
        "Tune",
        {
            'cs_devname': 'SI-Glob:DI-Tune',
            'cs_devtype': {CSDevTypes.TuneMeas, },
            'accelerator': 'SI',
            'sim_info': {'indices': [[]]},
            'cs_propties': {
                'tunex': {'name': '-H:TuneFrac-Mon'},
                'tuney': {'name': '-V:TuneFrac-Mon'}
            },
        }
    )


def define_bo(facil: Facility):
    """."""
    model = pymodels.bo.create_accelerator()
    facil.accelerators['BO'] = model
    # famdata = pymodels.bo.get_family_data(model)


def define_tb(facil: Facility):
    """."""
    model, _ = pymodels.tb.create_accelerator()
    facil.accelerators['TB'] = model
    # famdata = pymodels.tb.get_family_data(model)


def define_ts(facil: Facility):
    """."""
    model, _ = pymodels.ts.create_accelerator()
    facil.accelerators['TS'] = model
    # famdata = pymodels.ts.get_family_data(model)


def define_li(facil: Facility):
    """."""
    model, _ = pymodels.li.create_accelerator()
    facil.accelerators['LI'] = model
    # famdata = pymodels.li.get_family_data(model)


facility = Facility('sirius', 'epics', 'pyaccel')
facility.default_accelerator = 'SI'

define_si(facility)
define_bo(facility)
define_tb(facility)
define_ts(facility)
define_li(facility)


def get_lookup_table(table_name):
    magfunc = PSSearch.conv_pstype_2_magfunc(table_name.split('.')[0])
    magnet_conv_sign = -1
    if magfunc == 'corrector-horizontal':
        magnet_conv_sign = +1.0
    # if 'TB' in maname and 'QD' in maname:
    #     magnet_conv_sign = +1.0

    excdata = ExcitationData(table_name)
    cur = excdata.currents
    typ = excdata.main_multipole_type
    harm = excdata.main_multipole_harmonic
    intf = excdata.multipoles[typ][harm]

    table = np.array([cur, intf]).T
    table[:, 1] *= magnet_conv_sign
    return table


facility.get_lookup_table = get_lookup_table
