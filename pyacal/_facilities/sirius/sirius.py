"""Define SIRIUS facility object with aliases map."""

from copy import deepcopy as _dcopy

import numpy as _np
import pymodels

from ..facility import Facility

__CSDT = Facility.CSDevTypes


def define_si(facil: Facility):
    """."""
    model = pymodels.si.create_accelerator()
    facil.accelerators['SI'] = model

    famdata = pymodels.si.get_family_data(model)

    devname = famdata['DCCT']['devnames'][0]
    alias = devname.dev + devname.get_nickname()
    facil.add_2_alias_map(
        alias,
        {
            'cs_devname': devname,
            'cs_devtype': {__CSDT.DCCT, },
            'accelerator': 'SI',
            'sim_info': {
                'indices': famdata['DCCT']['index'][0],
            },
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
        alias = devname.dev + devname.get_nickname()
        facil.add_2_alias_map(
            alias,
            {
                'cs_devname': devname,
                'cs_devtype': {__CSDT.BPM, __CSDT.SOFB},
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
        )

    # --------- Define magnets (power supplies) ------------

    # ------------------ Slow Corrector Magnets ------------
    props = {
        'pwrstate_sp': {'name': ':PwrState-Sel'},
        'pwrstate_rb': {'name': ':PwrState-Sts'},
        'current_sp': {
            'name': ':Current-SP',
            'conv_sim2cs': {
                'excitation_table': 'name_of_excitation_table',
                'brho_source': '',
            }
        },
        'current_rb': {
            'name': ':CurrentRef-Mon',
            'conv_sim2cs': {
                'excitation_table': 'name_of_excitation_table',
                'brho_source': '',
            }
        },
        'current_mon': {
            'name': ':Current-Mon',
            'conv_sim2cs': {
                'excitation_table': 'name_of_excitation_table',
                'brho_source': '',
            }
        },
        'strength_sp': {
            'name': ':Current-SP',
            'conv_sim2cs': 1.0,
            'conv_hw2phys': {
                'exctitation_table': 'name_of_excitation_table',
                'brho_source': 'SI-Fam:PS-B-1:CurrentRef-Mon',
            }
        },
        'strength_rb': {
            'name': ':CurrentRef-Mon',
            'conv_cs2sim': 1.0,
            'conv_hw2phys': {
                'exctitation_table': 'name_of_excitation_table',
                'brho_source': 'SI-Fam:PS-B-1:CurrentRef-Mon',
            }
        },
        'strength_mon': {
            'name': ':Current-Mon',
            'conv_cs2sim': 1.0,
            'conv_hw2phys': {
                'exctitation_table': 'name_of_excitation_table',
                'brho_source': 'SI-Fam:PS-B-1:CurrentRef-Mon',
            }
        },
    }
    typs = ['CH', 'CV']
    typ_names = [
        __CSDT.CorrectorHorizontal,
        __CSDT.CorrectorVertical,
    ]
    for typ, name in zip(typs, typ_names):
        for i, idcs in enumerate(famdata[typ]['index']):
            devname = famdata[typ]['devnames'][i]
            alias = devname.dev + devname.get_nickname()
            facil.add_2_alias_map(
                alias,
                {
                    'cs_devname': devname,
                    'cs_devtype': {name, __CSDT.PowerSupply, __CSDT.SOFB},
                    'accelerator': 'SI',
                    'sim_info': {
                        'indices': idcs,
                    },
                    'cs_propties': _dcopy(props),
                }
            )

    # ----------------- Skew Quadrupoles --------------------------
    typ = 'QS'
    for i, idcs in enumerate(famdata[typ]['index']):
        devname = famdata[typ]['devnames'][i]
        alias = devname.dev + devname.get_nickname()
        facil.add_2_alias_map(
            alias,
            {
                'cs_devname': devname,
                'cs_devtype': {__CSDT.QuadrupoleSkew, __CSDT.PowerSupply},
                'accelerator': 'SI',
                'sim_info': {
                    'indices': idcs,
                },
                'cs_propties': _dcopy(props),
            }
        )

    # --------------------- Normal Quadrupoles -------------------
    props = {
        'pwrstate_sp': {'name': ':PwrState-Sel'},
        'pwrstate_rb': {'name': ':PwrState-Sts'},
        'current_sp': {
            'name': ':Current-SP',
            'conv_sim2cs': {
                'excitation_table': 'name_of_excitation_table',
                'brho_source': '',
                'companion_dev': '',
            }
        },
        'current_rb': {
            'name': ':CurrentRef-Mon',
            'conv_2sim2cs': {
                'excitation_table': 'name_of_excitation_table',
                'brho_source': '',
                'companion_dev': '',
            }
        },
        'current_mon': {
            'name': ':Current-Mon',
            'conv_sim2cs': {
                'excitation_table': 'name_of_excitation_table',
                'brho_source': '',
                'companion_dev': '',
            }
        },
        'strength_sp': {
            'name': ':Current-SP',
            'conv_sim2cs': 1.0,
            'conv_hw2phys': {
                'exctitation_table': 'name_of_excitation_table',
                'brho_source': 'SI-Fam:PS-B-1:CurrentRef-Mon',
                'companion_dev': '',
            }
        },
        'strength_rb': {
            'name': ':CurrentRef-Mon',
            'conv_cs2sim': 1.0,
            'conv_hw2phys': {
                'exctitation_table': 'name_of_excitation_table',
                'brho_source': 'SI-Fam:PS-B-1:CurrentRef-Mon',
                'companion_dev': '',
            }
        },
        'strength_mon': {
            'name': ':Current-Mon',
            'conv_cs2sim': 1.0,
            'conv_hw2phys': {
                'exctitation_table': 'name_of_excitation_table',
                'brho_source': 'SI-Fam:PS-B-1:CurrentRef-Mon',
                'companion_dev': '',
            }
        },
    }

    typ = 'QN'
    for i, idcs in enumerate(famdata[typ]['index']):
        devname = famdata[typ]['devnames'][i]
        alias = devname.dev + devname.get_nickname()
        facil.add_2_alias_map(
            alias,
            {
                'cs_devname': devname,
                'cs_devtype': {__CSDT.QuadrupoleNormal, __CSDT.PowerSupply},
                'accelerator': 'SI',
                'sim_info': {
                    'indices': idcs,
                },
                'cs_propties': _dcopy(props),
            }
        )

    # -------- Define RF --------
    facil.add_2_alias_map(
        "RFGen",
        {
            'cs_devname': 'RF-Gen',
            'cs_devtype': {__CSDT.RFGenerator, },
            'accelerator': 'SI',
            'sim_info': {
                'indices': famdata['SRFCav']['index'],
            },
            'cs_propties': {
                'frequency_rb': {
                    'name': ':GeneralFreq-RB', 'conv_cs2sim': 1,
                },
                'frequency_sp': {
                    'name': ':GeneralFreq-SP', 'conv_cs2sim': 1,
                },
            },
        }
    )

    facil.add_2_alias_map(
        "RFCav",
        {
            'cs_devname': 'RF-Gen',
            'cs_devtype': {__CSDT.RFCavity, },
            'accelerator': 'SI',
            'sim_info': {
                'indices': famdata['SRFCav']['index'],
            },
            'cs_propties': {
                'voltage_rb': {
                    'name': ':GeneralFreq-RB', 'conv_cs2sim': 1e6,
                },
                'voltage_sp': {
                    'name': ':GeneralFreq-SP', 'conv_cs2sim': 1e6,
                },
                'phase_rb': {
                    'name': ':GeneralFreq-RB', 'conv_cs2sim': _np.pi/180,
                },
                'phase_sp': {
                    'name': ':GeneralFreq-SP', 'conv_cs2sim': _np.pi/180,
                },
            },
        }
    )

    # -------- Define Tune Measurement Device --------
    facil.add_2_alias_map(
        "Tune",
        {
            'cs_devname': 'SI-Glob:DI-Tune',
            'cs_devtype': {__CSDT.TuneMeas, },
            'accelerator': 'SI',
            'sim_info': {
                'indices': [],
            },
            'cs_propties': {
                'tunex': {
                    'name': '-H:TuneFrac-Mon',
                },
                'tuney': {
                    'name': '-V:TuneFrac-Mon'
                }
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
