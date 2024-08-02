import at
import warnings
from .. import Facility
from ... import set_model

__CSDT = Facility.CSDevTypes

_DEVTYPE = {'CH': {__CSDT.CorrectorHorizontal, __CSDT.PowerSupply, __CSDT.SOFB},
            'CV': {__CSDT.CorrectorVertical, __CSDT.PowerSupply, __CSDT.SOFB},
            'QS': {__CSDT.QuadrupoleSkew, __CSDT.PowerSupply},
            'BPM': {__CSDT.BPM, __CSDT.SOFB},
            'BPM_ALL': {__CSDT.BPM, __CSDT.Family},
            'DCCT': {__CSDT.DCCT, },
            'RFGEN': {__CSDT.RFGenerator, },
            }

_DEVCONV = {'CH': 'hst',
            'CV': 'vst',
            'QS': 'sqp',
            }

def get_info_from_devname(devname):
    ds = devname.split('/')
    mag = ds[1].split('-')[1]
    cell, girder = ds[2].split('-')[:]
    return mag, cell, girder

def set_fcomp(devname, fcomp):
    ds = devname.split('/')
    ds1 = ds[1].split('-')
    return (ds[0]+'/'+_DEVCONV[fcomp]+'-'+ds1[1]+'/'+ds[2]).lower()

def define_ebs(facil:Facility):
    accname = 'EBS'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ring = at.load_lattice('./betamodel.mat', use='betamodel')
    set_model(accname, ring, facil)

    # Add SH correctors
    sh_idx = ring.get_uint32_index('SH*')
    properties = {'strength_read': {'name': 'Strength'},
                  'strength_write': {'name': 'Strength', 'wvalue': True},
                  'state': {'name': 'State'}}
    for idx in sh_idx:
        devname = ring[idx].Device
        mag, cell, girder = get_info_from_devname(devname)
        fcomp = ['CH', 'CV']
        for fc in fcomp:
            dname = set_fcomp(devname, fc)
            alias = (fc+'-'+mag+'-'+cell+'-'+girder).upper()
            facil.add_2_alias_map(
                alias,
                {'cs_devname': dname,
                 'cs_devtype': _DEVTYPE[fc],
                 'accelerator': accname,
                 'sim_info': {'indices': [[idx]], },
                 'cs_propties': properties,
                 }
            )

    # Add BPM
    bpm_idx = ring.get_uint32_index('BPM*')
    devname = 'srdiag/bpm/all'
    for i, idx in enumerate(bpm_idx):
        properties = {'posx': {'name': 'All_SA_HPosition', 'index': i},
                      'posy': {'name': 'All_SA_VPosition', 'index': i},
                      }
        alias = ring[idx].FamName
        facil.add_2_alias_map(
            alias,
            {'cs_devname': devname,
             'cs_devtype': _DEVTYPE['BPM'],
             'accelerator': accname,
             'sim_info': {'indices': [[idx]], },
             'cs_propties': properties,
            }
        )
    # Add BPM Family
    properties = {'orbx': {'name': 'All_SA_HPosition'},
                  'orby': {'name': 'All_SA_VPosition'},
                  }
    facil.add_2_alias_map(
        'BPM_ALL',
        {'cs_devname': devname,
         'cs_devtype': _DEVTYPE['BPM_ALL'],
         'accelerator': accname,
         'sim_info': {'indices': [bpm_idx], },
         'cs_propties': properties,
         }
    )


    # Add CT
    ct_idx = ring.get_uint32_index('*CT*')
    properties = {'current': {'name': 'Current',
                                  'conv_cs2sim': 1e3}, }
    devname = 'srdiag/beam-current/total'
    facil.add_2_alias_map(
        'DCCT',
        {'cs_devname': devname,
         'cs_devtype': _DEVTYPE['DCCT'],
         'accelerator': accname,
         'sim_info': {'indices': [ct_idx], },
         'cs_propties': properties,
         }
    )

    # Add RF Generator
    rf_idx = ring.get_uint32_index(at.RFCavity)
    properties = {'frequency_rb': {'name': 'Frequency'},
                  'frequency_sp': {'name': 'Frequency'},
                  }

    devname = 'sy/ms/1'
    alias = 'RFGEN'
    facil.add_2_alias_map(
        alias,
        {
            'cs_devname': devname,
            'cs_devtype': _DEVTYPE['RFGEN'],
            'accelerator': accname,
            'sim_info': {'indices': [rf_idx], },
            'cs_propties': properties,
        }
    )


# arbitrary, must be defined by the facility developers:
facility = Facility('esrf', 'tango', 'pyat')
facility.default_accelerator = 'EBS'
define_ebs(facility)
