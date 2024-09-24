# -*- coding: utf-8 -*-

from ..facility import Facility
import at

__CSDT = Facility.CSDevTypes

_DEVTYPE = {'CH': {__CSDT.CorrectorHorizontal, __CSDT.PowerSupply, __CSDT.SOFB},
            'CV': {__CSDT.CorrectorVertical, __CSDT.PowerSupply, __CSDT.SOFB},
            'QS': {__CSDT.QuadrupoleSkew, __CSDT.PowerSupply},
            'BPM': {__CSDT.BPM, __CSDT.SOFB},
            'BPM_ALL': {__CSDT.BPM, __CSDT.Family},
            'DCCT': {__CSDT.DCCT, },
            'RFGEN': {__CSDT.RFGenerator, },
            'Tune': {__CSDT.TuneMeas}
            }

#_DEVCONV = {'CH': 'hst',
#            'CV': 'vst',
#            'QS': 'sqp',
#            }


#%% Define the different machines

def define_storage_ring(facil: Facility):
    
    # --- Name of the ring ---
    accname = 'StorageRing'
    
    # --- Get the model and apply to the ring ---
    ring = at.load_lattice('./bessy2_standard_user.mat', use='THERING')
    facil.accelerators[accname] = ring
    
    # --- Add correctors ---
    
    # Find the correctors in the model
    hcorr = ring.get_uint32_index(at.checkattr('Corrector','H'))
    vcorr = ring.get_uint32_index(at.checkattr('Corrector','V'))
    
    # Define the PV suffix we are interested in. This is used together with
    # the device name to build up the full PV name
    properties = {'set': {'name': ':set'},
                   'read': {'name': ':rdbk'},
                   }  
    
    for idx in hcorr:      
        magnet_name = ring[idx].FamName
        # Get the power supply name since this is used for the PV name
        devname = magnet_name.replace('M', 'P')
        alias = magnet_name
        
        facil.add_2_alias_map(
            alias,
            {'cs_devname': devname,
             'cs_devtype': _DEVTYPE['CH'],
             'accelerator': accname,
             'sim_info': {'indices': [[idx]], },
             'cs_propties': properties,
             }
        )

    for idx in vcorr:      
        name = ring[idx].FamName
        # Get the power supply name since this is used for the PV name
        devname = name.replace('M', 'P')
        alias = name
        
        facil.add_2_alias_map(
            alias,
            {'cs_devname': devname,
             'cs_devtype': _DEVTYPE['CH'],
             'accelerator': accname,
             'sim_info': {'indices': [[idx]], },
             'cs_propties': properties,
             }
        )

    # --- Add BPMs ---    
    
    # Find the BPMs in the model
    bpm = ring.get_uint32_index('BPM*')
    
    # Define the PV suffix we are interested in. This is used together with
    # the device name to build up the full PV name
    properties = {'x_pos': {'name': ':rdX'},
                   'y_pos': {'name': ':rdY'},
                   } 
    
    for idx in bpm:
        name = ring[idx].FamName
        devname = name
        alias = name
        
        facil.add_2_alias_map(
            alias,
            {'cs_devname': devname,
             'cs_devtype': _DEVTYPE['BPM'],
             'accelerator': accname,
             'sim_info': {'indices': [[idx]], },
             'cs_propties': properties,
             }
        )            
        
    
    # # Add CT
    # ct_idx = ring.get_uint32_index('*CT*')
    # properties = {'current': {'name': 'Current',
    #                               'conv_cs2sim': 1e3}, }
    # devname = 'srdiag/beam-current/total'
    # facil.add_2_alias_map(
    #     'DCCT',
    #     {'cs_devname': devname,
    #      'cs_devtype': _DEVTYPE['DCCT'],
    #      'accelerator': accname,
    #      'sim_info': {'indices': [ct_idx], },
    #      'cs_propties': properties,
    #      }
    # )

    # # Add RF Generator
    # rf_idx = ring.get_uint32_index(at.RFCavity)
    # properties = {'frequency_rb': {'name': 'Frequency'},
    #               'frequency_sp': {'name': 'Frequency'},
    #               }

    # devname = 'sy/ms/1'
    # alias = 'RFGEN'
    # facil.add_2_alias_map(
    #     alias,
    #     {
    #         'cs_devname': devname,
    #         'cs_devtype': _DEVTYPE['RFGEN'],
    #         'accelerator': accname,
    #         'sim_info': {'indices': [rf_idx], },
    #         'cs_propties': properties,
    #     }
    # )
    # # -------- Define Tune Measurement Device --------
    # properties = {'tunex': {'name': 'Qh'},
    #               'tuney': {'name': 'Qv'},
    #               }

    # devname = 'srdiag/beam-tune/main'
    # alias = 'Tune'
    # facil.add_2_alias_map(
    #     alias,
    #     {
    #         'cs_devname': devname,
    #         'cs_devtype': _DEVTYPE['Tune'],
    #         'accelerator': accname,
    #         'sim_info': {'indices': [[]]},
    #         'cs_propties': properties
    #     }
    # )    
    
    
    pass

#%% Define which facility to use

facility = Facility('bessy2', 'epics', 'pyat')
facility.default_accelerator = 'StorageRing'
define_storage_ring(facility)
    
    
   
    
