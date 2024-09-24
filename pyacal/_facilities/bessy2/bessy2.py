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

    # --- Add BPMs ---       
    
    # for i, idcs in enumerate(famdata['BPM']['index']):
    #     devname = famdata['BPM']['devnames'][i]
    #     alias = devname.dev + '-' + devname.get_nickname()
    #     facil.add_2_alias_map(
    #         alias,
    #         {
    #             'cs_devname': devname,
    #             'cs_devtype': {CSDevTypes.BPM, CSDevTypes.SOFB},
    #             'accelerator': 'SI',
    #             'sim_info': {'indices': [idcs]},
    #             'cs_propties': {
    #                 'posx': {
    #                     'name': ':PosX-Mon',
    #                     'conv_cs2sim': 1e-9,  # from [nm] to [m]
    #                     'conv_cs2phys': 1e-3,  # from [nm] to [um]
    #                 },
    #                 'posy': {
    #                     'name': ':PosY-Mon',
    #                     'conv_cs2sim': 1e-9,  # from [nm] to [m]
    #                     'conv_cs2phys': 1e-3,  # from [nm] to [um]
    #                 },
    #             },
    #         }
    #     )
        
     
    # # Add BPM
    # bpm_idx = ring.get_uint32_index('BPM*')
    # devname = 'srdiag/bpm/all'
    # for i, idx in enumerate(bpm_idx):
    #     properties = {'posx': {'name': 'All_SA_HPosition', 'index': i},
    #                   'posy': {'name': 'All_SA_VPosition', 'index': i},
    #                   }
    #     alias = ring[idx].FamName
    #     facil.add_2_alias_map(
    #         alias,
    #         {'cs_devname': devname,
    #          'cs_devtype': _DEVTYPE['BPM'],
    #          'accelerator': accname,
    #          'sim_info': {'indices': [[idx]], },
    #          'cs_propties': properties,
    #         }
    #     )
    # # Add BPM Family
    # properties = {'orbx': {'name': 'All_SA_HPosition'},
    #               'orby': {'name': 'All_SA_VPosition'},
    #               }
    # facil.add_2_alias_map(
    #     'BPM_ALL',
    #     {'cs_devname': devname,
    #      'cs_devtype': _DEVTYPE['BPM_ALL'],
    #      'accelerator': accname,
    #      'sim_info': {'indices': [bpm_idx], },
    #      'cs_propties': properties,
    #      }
    # )


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
    
    
   
    
