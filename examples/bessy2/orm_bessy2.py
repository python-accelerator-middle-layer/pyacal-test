#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbit response matrix for BESSY II using pyacal
"""

#%% Import modules

import at
import numpy as np
import matplotlib.pyplot as plt
import pyacal
from pyacal.experiments.orbrespm import OrbRespm

#%% Set which machine to use 

pyacal.set_facility('bessy2')

#%% Set the model

ring = at.load_lattice('./bessy2_standard_user.mat', use='THERING')

pyacal.set_model('StorageRing', ring)

#%% Set the experiment to run

#orbrespm = OrbRespm(accelerator='EBS', isonline=True)

