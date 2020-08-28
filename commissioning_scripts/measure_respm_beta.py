#!/usr/bin/env python-sirius
"""."""

import time
import numpy as np
import pickle
from pymodels import si
from siriuspy.devices import SOFB, Tune, CurrInfo
from apsuite.commissioning_scripts.measure_beta import MeasBeta


def save_data(data, fname):
    """."""
    if not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'wb') as fil:
        pickle.dump(data, fil)


def create_devices():
    """."""
    sofb = SOFB(SOFB.DEVICES.SI)
    tune = Tune(Tune.DEVICES.SI)
    curr = CurrInfo(CurrInfo.DEVICES.SI)
    return sofb, tune, curr


def get_bpm_variation(sofb, period=10):
    """."""
    sofb.nr_points = 1
    orbx_pv = sofb.pv_object('SlowOrbX-Mon')
    orby_pv = sofb.pv_object('SlowOrbY-Mon')
    orbx_mon, orby_mon = [], []

    def orbx_cb(**kwargs):
        nonlocal orbx_mon
        orbx_mon.append(kwargs['value'])

    def orby_cb(**kwargs):
        nonlocal orby_mon
        orby_mon.append(kwargs['value'])

    orbx_pv.add_callback(orbx_cb)
    orby_pv.add_callback(orby_cb)
    time.sleep(period)
    orbx_pv.clear_callbacks()
    orby_pv.clear_callbacks()

    orbx = np.array(orbx_mon)
    orby = np.array(orby_mon)
    orb_var = np.array([np.std(orbx, axis=0), np.std(orby, axis=0)])
    return orb_var


def save_loco_setup(sofb, tune, curr, orbmat_name, fname):
    """."""
    loco_setup = dict()
    bpm_var = get_bpm_variation(sofb, period=10)
    loco_setup['timestamp'] = time.time()
    loco_setup['tunex'] = tune.tunex
    loco_setup['tuney'] = tune.tuney
    loco_setup['stored_current'] = curr.si.current
    loco_setup['sofb_nr_points'] = sofb.nr_points
    loco_setup['bpm_variation'] = bpm_var
    loco_setup['orbmat_name'] = orbmat_name
    save_data(data=loco_setup, fname=fname)


def measure_respm(sofb):
    """."""
    # Correct the orbit before the measurement
    wait_autocorr = 30
    sofb.cmd_turn_on_autocorr()
    time.sleep(wait_autocorr)

    # Turn off the correction loop
    sofb.cmd_turn_off_autocorr()

    # Start the respm measurement
    sofb.cmd_measrespmat()

    # Wait measurement to be finished
    sofb.wait_respm_meas()


def save_respmat_servconf(orbmat_name):
    """TO BE IMPLEMENTED."""


def measure_beta(model, famdata):
    """."""
    measbeta = MeasBeta(model, famdata)
    measbeta.params.nr_measures = 1
    measbeta.params.quad_deltakl = 0.01/2  # [1/m]
    measbeta.params.wait_quadrupole = 1  # [s]
    measbeta.params.wait_tune = 1  # [s]
    measbeta.params.timeout_quad_turnon = 10  # [s]
    measbeta.params.recover_tune = True
    measbeta.params.recover_tune_tol = 1e-4
    measbeta.params.recover_tune_maxiter = 5

    measbeta.params.quad_nrcycles = 0
    measbeta.params.time_wait_quad_cycle = 0.3  # [s]
    measbeta.quads2meas = list(measbeta.data['quadnames'])
    measbeta.start()
    wait_measbeta(measbeta)
    return measbeta


def wait_measbeta(measbeta, timeout=None):
    """."""
    timeout = timeout or 60 * 60
    interval = 1  # [s]
    ntrials = int(timeout/interval)
    for _ in range(ntrials):
        if not measbeta._thread.is_alive():
            break
        time.sleep(interval)
    else:
        print('WARN: Timed out waiting beta measurement.')


def run_respmat(nmeas):
    """."""
    orbmat_name = 'respmat_variation_n'
    setup_name = 'loco_setup_n'
    sofb, tune, curr = create_devices()
    for nms in range(nmeas):
        print('Respmat Measurement Number {:d}'.format(nms+1))
        orbname = orbmat_name + str(nms+1)
        filname = setup_name + str(nms+1)
        save_loco_setup(sofb, tune, curr, orbname, filname)
        measure_respm(sofb)
        save_respmat_servconf(orbname)


def run_beta(nmeas):
    """."""
    model = si.create_accelerator()
    famdata = si.get_family_data(model)
    beta_name = 'beta_variation_n'
    for nms in range(nmeas):
        print('Beta Measurement Number {:d}'.format(nms+1))
        bname = beta_name + str(nms+1)
        measbeta = measure_beta(model, famdata)
        measbeta.process_data(mode='pos')
        measbeta.save_data(bname)


def run_all(nmeas_respmat, nmeas_beta):
    """."""
    run_respmat(nmeas_respmat)
    run_beta(nmeas_beta)


run_all(nmeas_respmat=10, nmeas_beta=10)
