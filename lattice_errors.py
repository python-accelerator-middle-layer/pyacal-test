
import numpy as _np
import matplotlib.pyplot as _plt
import scipy.stats as _scystat

import pyaccel as _pyaccel
import lnls as _lnls

ERRORTYPES = ('x','y','roll','excit','k_dip')

def generate_errors(acc, mags=None, girder=None, fam_data=None,
        nr_mach=20, cutoff=1, rndtype='gauss', seed=42424242):
    """Generates random errors to be applied in the model by the function apply_errors.

    INPUTS:
      acc  : is the accelerator model.
      mags : list of dictionaries with keys:
        'labels' : list of family names.
        'nrsegs' : Optional. If existent and not None, must be a list of
          of the same length as 'labels', defining the number of segments
          the magnets of that family. It overwrites the fam_data option for
          the magtypes where it is defined.
        'x','y','roll','excit','k_dip' : errors definition (1-sigma for gauss,
          max for uniform)
      girder : dict with definition of girder errors. Possible keys: 'x','y','roll'.
      fam_data : dictionary whose keys are family names of all magnets
        defined in mags[i]['labels'] and the key 'girder' and values are
        dictionaries with at least one key: 'index', whose value is a list of
        indices the magnets in the lattice. If this list is a nested list, then
        each sub-list is understood as segments of the same physical magnet and
        the same error will be applied to them.
        Its default is None, which means each instance of the magnets in the
        lattice will be considered as independent with its own error, if 'nrsegs'
        is not defined for that magtype, and girder errors will be ignored.
        For sirius, this dictionary can be created with the function
        sirius.<'version'>.get_family_data(acc).
      nr_mach : generate errors for this number of machines.
      cutoff  : number of sigmas to truncate the distribution (default is 1)
      rndtype : type of distribution. Possible values: 'uniform' and 'gauss'.
                Default is 'gauss'.
      seed    : seed to generate random numbers. Default is 42424242

    OUTPUT:
      errors : dictionary with keys: 'x','y','roll','excit','k_dip'.
        Each key is a list with dimension nr_mach x len(acc)
        with errors generated for the elements defined by the inputs. If an
        element errors has contributions from 'mags' and 'girder', the value
        present in this output will be the sum of them.

    EXAMPLES:
     >>> acc = sirius.si.SI_V12.create_accelerator()
     >>> fam_data = sirius.SI_V12.get_family_data(acc)
     >>> um, mrad, percent = 1e-6, 1e-3, 1e-2
     >>> mags = []
     >>> dics = dict()
     >>> dics['labels'] = ['qfa','qdb2','qfb']
     >>> dics['x']     = 40 * um * 1
     >>> dics['y']     = 40 * um * 1
     >>> dics['roll']  = 0.20 * mrad * 1
     >>> dics['excit'] = 0.05 * percent * 1
     >>> mags.append(dics)
     >>> dics = dict()
     >>> dics['labels'] = ['b1','b2']
     >>> dics['x']     = 40 * um * 1
     >>> dics['y']     = 40 * um * 1
     >>> mags.append(dics)
     >>> girder = dict()
     >>> girder['x']     = 100 * um * 1
     >>> girder['y']     = 100 * um * 1
     >>> girder['roll']  =0.20 * mrad * 1
     >>> errors = generate_errors(acc,mags,girder,fam_data,nr_mach=20,cutoff=2)
    """

    _np.random.seed(seed=seed)
    #define the random numbers generator
    if rndtype.lower().startswith('gauss'):
        random_numbers = _scystat.truncnorm(-cutoff,cutoff).rvs
    elif rndtype.lower().startswith('unif'):
        random_numbers = _scystat.uniform(loc=-1,scale=2).rvs
    else:
        raise TypeError('Distribution type not recognized.')

    #generate empty arrays to store errors
    errors = dict()
    for err in ERRORTYPES:
        errors[err] = _np.zeros((nr_mach, len(acc)))

    if 'mags' is not None:
        for mtype in mags:
            ERRORS = [err for err in ERRORTYPES if err in mtype]
            for err in ERRORS:
                for ind,fam_name in enumerate(sorted(mtype['labels'])):
                    nrsegs = mtype.get('nrsegs',None)
                    if nrsegs is not None:
                        idx = _np.array(_pyaccel.lattice.find_indices(acc,'fam_name',fam_name))
                        idx = idx.reshape((-1,nrsegs[ind]))
                    elif fam_data is not None:
                        idx = fam_data[fam_name]['index']
                    else:
                        idx = _pyaccel.lattice.find_indices(acc,'fam_name',fam_name)
                    idx = _np.array(idx)
                    rnd = random_numbers((nr_mach,len(idx)))
                    if isinstance(idx[0],(list,tuple,_np.ndarray)):
                        rnd = rnd.repeat(len(idx[0]),axis=1)
                    errors[err][:,idx.ravel()] += rnd * mtype[err]

    if (girder is not None) and (fam_data is not None):
        ERRORS = [err for err in ERRORTYPES if err in girder]
        for err in ERRORS:
            for gir in fam_data['girder']:
                idx = _np.array(gir['index'])
                rnd = random_numbers((nr_mach,1)).repeat(len(idx),axis=1)
                errors[err][:,idx.ravel()] += rnd * girder[err]

    return errors

def apply_erros(machine, errors, increment=1.0):
    """Apply the errors generated by generate_errors to the ring model.

    INPUTS:
      machine  : might be a model of the ring or a list of models of
                 the ring
      errors   : structure of errors to be applied (for more details see
                 generate_errors help
      increment: float defining the fraction of the errors which will be
                 additively applied to the machines.

    OUTPUT:
      machine  : lis of ring models with errors.
    """
    def apply_errors_one_machine(ring, errors, ii, fraction):
        funs={'x':_pyaccel.lattice.add_error_misalignment_x,
              'y':_pyaccel.lattice.add_error_misalignment_y,
              'roll':_pyaccel.lattice.add_error_rotation_roll,
              'excit':_pyaccel.lattice.add_error_excitation_main,
              'k_dip':_pyaccel.lattice.add_error_excitation_kdip}
            #   'yaw':_pyaccel.lattice.add_error_rotation_yaw,
            #   'pitch':_pyaccel.lattice.add_error_rotation_pitch,

        for errtype in errors:
            err = fraction * errors[errtype][ii,:]
            idx, *_ = err.nonzero()
            funs[errtype](ring, idx, err[idx])

    nr_mach = errors['x'].shape[0]

    machs = []
    if not isinstance(machine,(list,tuple)):
        for i in range(nr_mach):
            machs.append(_pyaccel.accelerator.Accelerator(accelerator=machine))
        machine = machs

    if len(machine) != nr_mach:
        print('DifferentSizes: Incompatibility between errors and'+
                ' machine lengths.\n Using minimum of both.')
        nr_mach = min([len(machine),nr_mach])

    ids_idx  = [i for i in range(len(machine[0]))
                  if machine[0][i].pass_method.startswith('kicktable_pass')]
    sext_idx = [i for i in range(len(machine[0]))
                  if machine[0][i].polynom_b[2] != 0.0]

    print('    ------------------------------- ')
    print('   |   codx [mm]   |   cody [mm]   |')
    print('   | (max)   (rms) | (max)   (rms) |')
    print('---|-------------------------------|')
   #print('001| 13.41   14.32 | 13.41   14.32 |');
    for i in range(nr_mach):
        apply_errors_one_machine(machine[i], errors, i, increment)
        ring = machine[i][:]
        for ii in ids_idx:  ring[ii].pass_method = 'drift_pass'
        for ii in sext_idx: ring[ii].polynom_b[2] = 0.0
        codx, cody = _calc_cod(ring)
        x_max_all, x_rms_all = 1e3*_np.abs(codx).max(), 1e3*codx.std(ddof=1)
        y_max_all, y_rms_all =  1e3*_np.abs(cody).max(), 1e3*cody.std(ddof=1)
        print('{0:03d}| {1:5.2f}   {2:5.2f} | {3:5.2f}   {4:5.2f} |'.format(
                                    i,x_max_all,x_rms_all,y_max_all,y_rms_all))
    print(36*'-')
    return machine

def correct_cod(machine, bpms, hcms, vcms, sext_ramp=[1.0], svs='all',tol=1e-5,
    nr_iter=20, ind_bba=None, bpm_err=None, respm=None, gcodx=None, gcody=None):
    """  Correct orbit of several machines.

     INPUTS:
       machine  : list of lattice models to correct the orbit.
       gcodx    : horizontal reference orbit to use in correction. May be a list
          defining the orbit for each bpm. In this case the reference will be the
          same for all the machines. Or Can be an array or nested list with dimension
          nr_machines X nr_bpms, to define different orbits among the machines.
          If not passed a default of zero will be used.
       gcody    : same as goal_codx but for the vertical plane.
       bpms - Indices of the bpms in the model.
       hcms - Indices of horizontal correctors in the model. If the correctors
             are segmented in the model, it must be a nested list or a 2D array
             where each sub-list or first dimension of the array have the
             indices of one corrector.
       vcms - Indices of vertical correctors in the model. The same as 'hcms'
             for segmented correctors.
       ### All keys below are optional ###
       sext_ramp - If existent, must be a vector with components less than or
             equal to one, denoting a fraction of sextupoles strengths used in each step
             of the correction. For example, if sext_ramp = [0,1] the correction
             algorithm will be called two times for each machine. In the first
             time the sextupoles strengths will be zeroed and in the second
             time they will be set to their correct value. If the last value is
             not 1, this value will be appended. Default: [1.0].
       svs - may be a number denoting how many singular values will be
             used in the correction or the string 'all' to use all singular
             values. Default: 'all';
       nr_iter - Optional. Maximum number of iterations the correction
             algortithm will perform at each call for each machine. Default: 20
       tol - Optional. If existent must be a float defining the threshold
             for convergence of the algorithm. By convergence we mean the
             relative variation of the error function between two successive
             iterations. Default: 1e-5.
       ind_bba - Optional. If not None, must be a list or 1D numpy
             array with the indices of elements of the accelerator with the same
             length as the bpms. The algorithm will take the misalignment error
             of those elements, add them to the gcod and correct the orbit
             to this reference. For example, the function get_bba_ind returns
             the indices of the nearest quadrupoles to each bpm. If those indices
             were passed to this function, the orbit would be corrected to their
             magnetic center, simulating a BBA prcedure.
       bpm_err - Optional. If not None, must be a dictionary with a mandatory
             key: 'sigma' which is a tuple of length 2 with the
             rms of the noise in the horizontal and vertical plane, respectively
             and an optional key: 'cutoff' which must be a float defining in how
             many sigma the distribution must be truncated (default: 1). This
             noise will be added to the reference orbit (gcod + bba) simulating
             errors in the bpms readings (offset errors, bba precision,...).
       respm - dictionary with keys 'M', 's', 'V', 'U' which are the response
             matrix and its SVD decomposition (M = U * diag(s) * V). If None,
             the function WILL CALCULATE the response matrix for each machine in
             each step defined by sext_ramp.

    OUTPUT:
      gcodx : 2D numpy array with the reference orbit, excluding the bpm noise
         (gcodx + bba), used in the correction.
      gcody : the same as gcodx, but for the vertical plane.
    """
    nr_mach = len(machine)

    # making sure they are in order
    bpms = sorted(bpms)
    hcms = sorted(hcms)
    vcms = sorted(vcms)

    #### Dealing with optional input parameters:
    if svs == 'all': svs = len(hcms)+len(vcms)
    if not _np.allclose(sext_ramp[-1],1,atol=1e-8,rtol=1e-8): sext_ramp.append(1)
    if bpm_err is not None:
        cutoff = bpm_err.get('cutoff',1)
        sigs   = bpm_err.pop('sigma')
        noisex = sigs[0]*_scystat.truncnorm(-cutoff,cutoff).rvs((nr_mach,len(bpms)))
        noisey = sigs[1]*_scystat.truncnorm(-cutoff,cutoff).rvs((nr_mach,len(bpms)))
    else:
        noisex = _np.zeros((nr_mach,len(bpms)))
        noisey = _np.zeros((nr_mach,len(bpms)))

    # hcms must behave as if the magnet was segmented:
    types = (list,tuple,_np.ndarray)
    if not isinstance(hcms[0],types): hcms = [[ind] for ind in hcms]
    if not isinstance(vcms[0],types): vcms = [[ind] for ind in vcms]
    #goal orbit must be a numpy array for each machine
    if gcodx is None:     gcodx = _np.zeros((nr_mach,len(bpms)))
    elif len(gcodx) == 1: gcodx = _np.array([gcodx]).repeat(nr_mach,axis=0)
    if gcody is None:     gcody = _np.zeros((nr_mach,len(bpms)))
    elif len(gcody) == 1: gcody = _np.array([gcodx]).repeat(nr_mach,axis=0)
    ####

    print('correcting closed-orbit distortions')
    print('sextupole ramp: {0}'.format(sext_ramp))
    print('selection of singular values: {0:3d}'.format(svs))
    print('maximum number of orbit correction iterations: {0:3d}'.format(nr_iter))
    print('tolerance: {0:8.2e}\n'.format(tol))
    print('    -----------------------------------------------------------------------------------------------')
    print('   |           codx [um]           |           cody [um]           |  kickx[urad]     kicky[urad]  | (nr_iter|nr_refactor)')
    print('   |      all             bpm      |      all             bpm      |                               | [sextupole ramp]')
    print('   | (max)   (rms) | (max)   (rms) | (max)   (rms) | (max)   (rms) | (max)   (rms) | (max)   (rms) | ',end='')
    print(' '.join(['{0:7.5f}'.format(ind) for ind in sext_ramp]))
    print('\n---|---------------------------------------------------------------|-------------------------------| ')


    #Definition of nonlinear elements indices and ids indices:
    sext_idx = [ind for ind in range(len(machine[0])) if machine[0][ind].polynom_b[2] != 0]
    ids_idx  = [ind for ind in range(len(machine[0]))
                    if machine[0][ind].pass_method.startswith('kicktable_pass')]

    #Definition of Stats functions:
    s = _pyaccel.lattice.find_spos(machine[0])
    max_rms  = lambda v: (1e6*_np.abs(v).max(), 1e6*_np.array(v).std(ddof=1))
    max_rmss = lambda v: (1e6*_np.abs(v).max(), 1e6*_np.sqrt((_np.trapz(v**2,x=s)
                                                             -_np.trapz(v,x=s)**2/s[-1])/s[-1]))
    for i in range(nr_mach):
        sext_str = [machine[i][ind].polynom_b[2] for ind in sext_idx]

        if ind_bba is not None:
            gcodx[i] += _pyaccel.lattice.get_error_misalignment_x(machine[i],ind_bba)
            gcody[i] += _pyaccel.lattice.get_error_misalignment_y(machine[i],ind_bba)

        Tgcodx = gcodx[i] + noisex[i]
        Tgcody = gcody[i] + noisey[i]

        niter  = _np.zeros(len(sext_ramp),dtype=int)
        ntimes = _np.zeros(len(sext_ramp),dtype=int)
        for ind in ids_idx: machine[i][ind].pass_method = 'drift_pass'
        for j in range(len(sext_ramp)):
            if j == len(sext_ramp)-1:
                for ind in ids_idx: machine[i][ind].pass_method = 'kicktable_pass'

            for ii,ind in enumerate(sext_idx): machine[i][ind].polynom_b[2] = sext_str[ii]*sext_ramp[j]

            hkck,vkck,codx,cody,niter[j],ntimes[j] = cod_sg(machine[i],bpms,hcms,vcms,
                respm=respm,nr_iters=nr_iter,svs=svs,tolerance=tol,gcodx=Tgcodx,gcody=Tgcody)
            if any([_np.isnan(codx).any(),_np.isnan(cody).any()]):
                print('Machine {0:03d} became unstable when sextupole strength was {1:5.3f}'.format(i,sext_ramp[j]))
                for ii,ind in enumerate(sext_idx): machine[i][ind].polynom_b[2] = sext_str[ii]
                break
        print('{0:03d}|'.format(i)+
              ' {0:5.1f}   {1:5.1f} |'.format(*max_rmss(codx))+
              ' {0:5.1f}   {1:5.1f} |'.format(*max_rms(codx[bpms] - gcodx[i]))+
              ' {0:5.1f}   {1:5.1f} |'.format(*max_rmss(cody))+
              ' {0:5.1f}   {1:5.1f} |'.format(*max_rms(cody[bpms] - gcody[i]))+
              ' {0:5.1f}   {1:5.1f} |'.format(*max_rms(hkck))+
              ' {0:5.1f}   {1:5.1f} |'.format(*max_rms(vkck)), end='')
        print(' '.join(['({0:02d}|{1:02d})'.format(niter[ind], ntimes[ind])
                        for ind in range(len(niter))]))
    print('--------------------------------------------------------------------------------------------------- \n');
    return gcodx, gcody

def calc_respm_cod(acc,bpm_idx,hcm_idx,vcm_idx,symmetry=1,printing=False):
    """

    """
    # making sure they are in order
    bpm_idx = sorted(bpm_idx)
    hcm_idx = sorted(hcm_idx)
    vcm_idx = sorted(vcm_idx)
    nr_bpms = len(bpm_idx)
    nr_hcms = len(hcm_idx)
    nr_vcms = len(vcm_idx)
    M = _np.zeros((2*nr_bpms,nr_vcms+nr_hcms)) # create response matrix and its components
    Mxx, Mxy = M[:nr_bpms,:nr_hcms], M[:nr_bpms,nr_hcms:]
    Myx, Myy = M[nr_bpms:,:nr_hcms], M[nr_bpms:,nr_hcms:]

    len_bpm = nr_bpms // symmetry
    len_hcm = nr_hcms // symmetry
    len_vcm = nr_vcms // symmetry
    if (len_bpm % 1) or (len_hcm % 1) or (len_vcm % 1):
        len_bpm = nr_bpm
        len_hcm = nr_hcm
        len_vcm = nr_vcm
        symmetry = 1
    else:
        hcm_idx = hcm_idx[:len_hcm]
        vcm_idx = vcm_idx[:len_vcm]

    if printing:
        print('bpms:{0:03d}, hcms:{0:03d}, vcms:{0:03d}'.format(
                   nr_bpms, nr_hcms, nr_vcms))

    mxx,mxy,myx,myy = _get_response_matrix(acc, bpm_idx, hcm_idx, vcm_idx)

    for i in range(symmetry):
        indcs = list(range(i*len_hcm,(i+1)*len_hcm))
        Mxx[:,indcs] = _np.roll(mxx,len_bpm*i,axis=0)
        Myx[:,indcs] = _np.roll(myx,len_bpm*i,axis=0)

        indcs = list(range(i*len_vcm,(i+1)*len_vcm))
        Mxy[:,indcs] = _np.roll(mxy,len_bpm*i,axis=0) #the last bpm turns into the first
        Myy[:,indcs] = _np.roll(myy,len_bpm*i,axis=0)

    r = dict()
    r['M'] = M

    U, s, V = _np.linalg.svd(M,full_matrices=False) #M = U*np.diag(s)*V
    r['U'] = U
    r['V'] = V
    r['s'] = s

    if printing:
        print('number of singular values: {0:03d}'.format(len(s)))
        print('singular values: {0:3f},{0:3f},{0:3f} ... {0:3f},{0:3f},{0:3f}'.format(
                                 s[0],  s[1],  s[2],      s[-3],s[-2],s[-1]))
    return r

def cod_sg(acc,bpms,hcms,vcms,respm=None, nr_iters=20,svs='all',tolerance=1e-5,
                    gcodx=None,gcody=None):

    if gcodx is None: gcodx = _np.zeros(bpms.shape)
    if gcody is None: gcody = _np.zeros(bpms.shape)

    if respm is None:
        respm = calc_respm_cod(acc, bpms, hcms, vcms, symmetry=1, printing=False)
    s = respm['s']
    U = respm['U']
    V = respm['V']
    #selection of singular values
    if svs == 'all': svs = len(hcms)+len(vcms)
    invs  = 1/s
    invs[svs:] = 0
    iS = _np.diag(invs)
    CM = - _np.dot(V.T,_np.dot(iS,U.T))

    gcod = _np.zeros(2*len(bpms))
    gcod[:len(bpms)] = gcodx
    gcod[len(bpms):] = gcody

    cod  = _np.zeros(2*len(bpms))
    cod[:len(bpms)], cod[len(bpms):] = _calc_cod(acc, indices=bpms)

    corrs = _lnls.utils.flatten(hcms)
    corrs.extend(_lnls.utils.flatten(vcms))
    corrs = _np.unique(_np.array(corrs))
    best_fm = (cod - gcod).std(ddof=1)
    best_acc = acc[corrs]
    factor, ntimes = 1, 0
    for iters in range(nr_iters):
        # calcs kicks
        cod[:len(bpms)], cod[len(bpms):] = _calc_cod(acc, indices=bpms)

        dkicks = factor * _np.dot(CM, cod - gcod)
        # sets kicks
        hkicks = dkicks[:len(hcms)]
        vkicks = dkicks[len(hcms):]
        hkicks += _get_kickangle(acc, hcms, 'x')
        vkicks += _get_kickangle(acc, vcms, 'y')
        _set_kickangle(acc, hcms, hkicks, 'x')
        _set_kickangle(acc, vcms, vkicks, 'y')
        cod[:len(bpms)], cod[len(bpms):] = _calc_cod(acc, indices=bpms)

        fm = (cod - gcod).std(ddof=1)
        residue = abs(best_fm-fm)/best_fm
        if fm < best_fm:
            best_fm   = fm
            best_acc  = acc[corrs]
            factor    = 1 # reset the correction strength to 1
        else:
            acc[corrs] = best_acc
            factor *= 0.75 # reduces the strength of the correction
            ntimes += 1    # check how many times it passed here
        # breaks the loop in case convergence is reached
        if residue < abs(tolerance): break
    hkicks = _get_kickangle(acc, hcms, 'x');
    vkicks = _get_kickangle(acc, vcms, 'y');
    codx, cody = _calc_cod(acc)
    return hkicks, vkicks, codx, cody, iters+1, ntimes

def get_bba_ind(acc,bpms):

    # Determinando indices dos quadrupolos na rede
    quad_idx = _np.array([ind for ind in range(len(acc)) if acc[ind].angle == 0 and acc[ind].K != 0])
    #descobrindo qual a posicao dos elementos na rede
    spos = _pyaccel.lattice.find_spos(acc,indices='open')
    bpm_spos = spos[bpms] # dos bpms
    quad_spos = spos[quad_idx] # e dos quadrupolos
    #determinando quais sao os quadrupolos mais proximos aos bpms
    In = _np.abs(bpm_spos[:,None] - quad_spos[None,:]).argmin(axis=1)
    ind_bba = quad_idx[In]
    return ind_bba

def correct_coupling(machine, bpms, hcms, vcms, scms, svs='all',nr_iter=20,
    tol=1e-5, bpm_err=None, respm=None):
    """ Correct coupling of several machines.

     INPUTS:
       name     : name of the file to which the inputs will be saved;
       machine  : cell array of lattice models to symmetrize the optics.
       coup     : structure with fields:
          bpm_idx   - bpm indexes in the model;
          hcm_idx   - horizontal correctors indexes in the model;
          vcm_idx   - vertical correctors indexes in the model;
          scm_idx   - indexes of the skew quads which will be used to symmetrize;
          svs       - may be a number denoting how many singular values will be
             used in the correction or the string 'all' to use all singular
             values. Default: 'all';
          max_nr_iter - maximum number of iteractions the correction
             algortithm will perform at each call for each machine;
          tolerance - if in two subsequent iteractions the relative difference
             between the error function values is less than this value the
             correction is considered to have converged and will terminate.
          simul_bpm_corr_err - if true, the Gains field defined in the bpms  and
             the Gain field defined in the correctors in thelattice will be used
             to simulate gain errors in these elements, changing the response
             matrix calculated. Notice that the supra cited fields must exist
             in the lattice models of the machine array for each bpm and corrector
             in order for this this simulation to work. Otherwise an error will occur.
          respm - structure with fields M, S, V, U which are the coupling response
             matrix and its SVD decomposition. If NOT present, the function
             WILL CALCULATE the coupling response matrix for each machine.

     OUTPUT:
       machine : cell array of lattice models with the orbit corrected.
    """

    bpms = sorted(bpms)
    hcms = sorted(hcms)
    vcms = sorted(vcms)
    scms = sorted(scms)
    nr_mach = len(machine)

    if svs == 'all': svs = len(hcms)+len(vcms)
    if bpm_err is not None:
        cutoff = bpm_err.get('cutoff',1)
        sigs   = bpm_err.pop('sigma')
        noisex = sigs[0]*_scystat.truncnorm(-cutoff,cutoff).rvs((nr_mach,len(bpms)))
        noisey = sigs[1]*_scystat.truncnorm(-cutoff,cutoff).rvs((nr_mach,len(bpms)))
    else:
        noisex = _np.zeros((nr_mach,len(bpms)))
        noisey = _np.zeros((nr_mach,len(bpms)))

    # all correctors must behave as if the magnet was segmented:
    types = (list,tuple,_np.ndarray)
    if not isinstance(hcms[0],types): hcms = [[ind] for ind in hcms]
    if not isinstance(vcms[0],types): vcms = [[ind] for ind in vcms]
    if not isinstance(scms[0],types): scms = [[ind] for ind in scms]

    print('correcting coupling (minimization of non-diagonal response matrix)')
    print('selection of singular values: {0:3d}'.format(svs))
    print('maximum number correction iterations: {0:3d}'.format(nr_iter))
    print('tolerance: {0:8.2e}\n'.format(tol))
    print('     ------------------------------------------------------- ')
    print('    | Max Kl |  chi2  |      Coup[%]     | NIters | NRedStr |')
    print('    | [1/km] |        |  Ey/Ex  | Dy[mm] |        |         |')
    print('------------------------------------------------------------|')

    #Definition of Stats functions:
    s = _pyaccel.lattice.find_spos(machine[0])
    mom2s = lambda v: 1e6*_np.sqrt(_np.trapz(v**2,x=s)/s[-1])
    for i in range(nr_mach):
        RTr = _pyaccel.tracking.calc_emittance_coupling(machine[i])
        twiss, *_ = calc_twiss(accelerator=machine[i], indices = 'closed')
        D = twiss.etax

        skewstr, iniFM, bestFM, niter, n_times = coup_sg(machine[i],bpms,hcms,vcms,
            scms, respm=respm,nr_iters=nr_iter,svs=svs,tolerance=tol)

        RTr2 = _pyaccel.tracking.calc_emittance_coupling(machine[i])
        twiss, *_ = calc_twiss(accelerator=machine[i], indices = 'closed')
        D2 = twiss.etax

        print('{0:03d} | {1:6s} | {2:6.3f} | {3:7.3f}  | {4:6.3f} |  {5:4s}  |  {6:4s}   |'.format(
            i, ' ', iniFM, 100*RTr, mom2s(D), ' ',' '))
        print('{0:3s} | {1:6.2f} | {2:6.3f} | {3:7.3f}  | {4:6.3f} |  {5:4s}  |  {6:4s}   |'.format(
            ' ', 1000*_np.abs(skewstr).max(), bestFM, 100*RTr2, mom2s(D2), niter, n_times))
        print('------------------------------------------------------------|')

def calc_respm_coupling(acc, bpms, hcms, vcms, scms, symmetry=1, info=None):
    bpms = sorted(bpms)
    hcms = sorted(hcms)
    vcms = sorted(vcms)
    scms = sorted(scms)

    if info is None:
        stepK0 = 0.001;

        print('-  collecting info for coupling respm calculation ...')
        print('   (this routine is yet to be generalized for arbitrary segmented skew quadrupole models!)')
        print('   qs:{0:03d}'.format(len(scms)))

        # Test hysteresis
        hyster = 0.0;
        stepK = stepK0*(1-hyster)# Test hysteresis

        len_scm = len(scms) // symmetry
        len_bpm = len(bpms) // symmetry
        len_hcm = len(hcms) // symmetry
        len_vcm = len(vcms) // symmetry
        if len_scms % 1:
            len_scm = len_scm*symmetry
            len_bpm = len_bpm*symmetry
            len_hcm = len_hcm*symmetry
            len_vcm = len_vcm*symmetry
            symmetry = 1

        # this routine has to be generalized for arbitrary skew quad segmented models !!!
        info = []
        for i in range(len_scm):
            for ii in scms[i]: acc[ii].Ks += stepK/2
            Mp, Dispp, tunep = _get_matrix_disp(acc, bpms, hcms, vcms)
            for ii in scms[i]: acc[ii].Ks += -stepK
            Mn, Dispn, tunen = _get_matrix_disp(acc, bpms, hcms, vcms)
            for ii in scms[i]: acc[ii].Ks += +stepK/2
            info.append({'M':(Mp-Mn)/stepK0,
                         'D':(Dispp-Dispn)/stepK0,
                         'Tune':(tunep-tunen)/stepK0})

        if symmetry != 1:
            for i in range(len_scm):
                Mx, My   = _np.vsplit(info[i]['M'],2)
                Mxx, Mxy = _np.hsplit(Mx,[symmetry*len_hcm])
                Myx, Myy = _np.hsplit(My,[symmetry*len_hcm])
                Disp = info[i].['D']
                for ii in range(1,symmetry-1):
                    Mxx = _np.roll(_np.roll(Mxx,len_bpm,axis=0),len_hcm,axis=1)
                    Myx = _np.roll(_np.roll(Myx,len_bpm,axis=0),len_hcm,axis=1)
                    Mxy = _np.roll(_np.roll(Mxy,len_bpm,axis=0),len_vcm,axis=1)
                    Myy = _np.roll(_np.roll(Myy,len_bpm,axis=0),len_vcm,axis=1)
                    Disp= _np.roll(Disp,len_bpm,axis=0)
                    Mx = _np.hstack((Mxx,Mxy))
                    My = _np.hstack((Myx,Myy))
                    info[i+len_scm*ii]['M']    = _np.vstack((Mx,My))
                    info[i+len_scm*ii]['D']    = Disp
                    info[i+len_scm*ii]['Tune'] = info[i]['Tune']

    _, Mxy, Myx, _, _, Dispy = _prepare_data_for_symm(acc, coup, info[0]['M'], info[0]['D'])
    v = _calc_residue_coupling(Mxy, Myx, Dispy, bpms, hcms, vcms)
    M = _np.zeros((len(v),len(info)))
    M[:,0] = v
    for i = range(1,len(info)):
        _, Mxy, Myx, _, _, Dispy = _prepare_data_for_symm(acc, coup, info[i]['M'], info[i]['D'])
        M[:,i] = _calc_residue_coupling(Mxy, Myx, Dispy, bpms, hcms, vcms)

    r['M'] = M
    U, s, V = _np.linalg.svd(M,full_matrices=False) #M = U*np.diag(s)*V
    r['U'] = U
    r['V'] = V
    r['S'] = s

    print('   number of singular values: %03i\n', len(s))
    print('   singular values: %f,%f,%f ... %f,%f,%f\n', s[0],s[1],s[2],s[-3],s[-2],s[-1])
    return respm, info

def coup_sg(acc, bpms, hcms, vcms, scms, respm=None, svs='all', nr_iter=20,
    tol=1e-5, bpm_err=None):

    def calc_residue_for_optimization(accel):
        M, Disp, _ = _get_matrix_disp(accel, bpms, hcms, vcms)
        [~, Mxy, Myx, ~, ~, Dispy] = _prepare_data_for_symm(accel, coup, M, Disp)
        return _calc_residue_coupling(Mxy, Myx, Dispy, bpms, hcms, vcms)

    bpms = sorted(bpms)
    hcms = sorted(hcms)
    vcms = sorted(vcms)
    scms = sorted(scms)

    if respm is None:
        respm = calc_respm_coup(acc, bpms, hcms, vcms, scms, symmetry=1, info=None)
    s = respm['s']
    U = respm['U']
    V = respm['V']
    #selection of singular values
    if svs == 'all': svs = len(scms)
    invs  = 1/s
    invs[svs:] = 0
    iS = _np.diag(invs)
    CM = - _np.dot(V.T,_np.dot(iS,U.T))

    skews = _lnls.utils.flatten(scms)
    skews = _np.unique(_np.array(skews))

    best_coupvec = calc_residue_for_optimization(acc)
    best_skew    = acc[skews]
    best_fm = best_coupvec.std(ddof=1)
    init_fm = best_fm
    factor, n_times = 1, 0
    for niter = range(nr_iter):
        # calcs kicks:
        kicks = factor * _np.dot(CM, best_coupvec)
        # set the kicks:
        for i in range(len(kicks)):
            for ii in range(len(scms[i])):
                acc[scms[i][ii]].Ks += kicks[i]

        coup_vec = calc_residue_for_optimization(acc)
        fm = coup_vec.std(ddof=1)
        residue = abs(best_fm-fm)/best_fm
        if fm < best_fm:
            best_fm      = fm
            best_skew    = acc[skews]
            factor = 1 # reset the correction strength to 1
            best_coupvec  = coup_vec
        else:
            acc[skews] = best_skew
            factor  *= 0.75 # reduces the strength of the correction
            n_times += 1    # to check how many times it passed here;
        # breaks the loop in case convergence is reached
        if residue < abs(tol): break

    # get the kick strength:
    skewstr = _np.array(len(scms),dtype=float)
    for i in range(len(scms)):
        for ii in range(len(scms[i])):
            skewtr[i] += acc[scms[i][ii]].Ks * acc[scms[i][ii]].length

    return skewstr, init_fm, best_fm, niter, n_times

def _calc_residue_coupling(Mxy, Myx, Dispy, bpms, hcms, vcms):
    disp_weight = (len(hcms) + len(vcms))*10
    v = _np.hstack((Myx, Mxy, disp_weight * Dispy.T)) # para ficar ordenado por bpm
    return v.flatten()

def _prepare_data_for_symm(the_ring, optics, M, Disp): return None
    # # assumes uniform dipolar field for orbit correctors
    #
    # len_bpms = size(optics.bpm_idx,1);
    # if optics.simul_bpm_corr_err
    #     bpm_gains = getcellstruct(the_ring,'Gains',optics.bpm_idx(:,1));
    #     hcm_gains = getcellstruct(the_ring,'Gain',optics.hcm_idx(:,1))';
    #     vcm_gains = getcellstruct(the_ring,'Gain',optics.vcm_idx(:,1))';
    #     M = repmat([hcm_gains,vcm_gains],size(M,1),1).*M;
    #     for i=1:len(bpm_gains)
    #         M(i+[0,len_bpms],:) = bpm_gains{i}*M(i+[0,len_bpms],:);
    #         Disp([1,3],i) = bpm_gains{i}*Disp([1,3],i);
    #
    # M = mat2cell(M,len_bpms*[1,1],[size(optics.hcm_idx,1), size(optics.vcm_idx,1)]);
    # Mxx = M{1,1};
    # Mxy = M{1,2};
    # Myx = M{2,1};
    # Myy = M{2,2};
    #
    # Dispx = Disp(1,:);
    # Dispy = Disp(3,:);
    # return Mxx,Mxy,Myx,Myy, Dispx, Dispy

def _get_matrix_disp(acc, bpms, hcms, vcms):
    mxx,mxy,myx,myy = _get_response_matrix(acc, bpms, hcms, vcms)
    M  = _np.vstack((_np.hstack((mxx,mxy)), _np.hstack((myx,myy))))
    twiss = _pyaccel.optics.calc_twiss(acc)
    Disp = [twiss.etax, twiss.etay]
    tune = _np.array([twiss.mux[-1],twiss.muy[-1]])/2/_np.pi
    return M, Disp, tune

def _calc_cod(acc, indices = 'open'):
    if acc.cavity_on:
        orb = _pyaccel.tracking.findorbit6(acc,indices=indices)
    else:
        orb = _pyaccel.tracking.findorbit4(acc,indices=indices)
    return orb[0],orb[2]

def _get_response_matrix(acc, bpms, hcms, vcms):
    # M(y,x) --> y : orbit    x: corrector

    if not isinstance(hcms[0],(list,tuple,_np.ndarray)):
        hcms = [[ind] for ind in hcms]
    if not isinstance(vcms[0],(list,tuple,_np.ndarray)):
        vcms = [[ind] for ind in vcms]

    closed_orbit = _np.zeros((6,len(acc)))
    if acc.cavity_on:
        closed_orbit = _pyaccel.tracking.findorbit6(acc,indices='open')
        M, T = _pyaccel.tracking.findm66(acc,closed_orbit=closed_orbit)
    else:
        closed_orbit[:4,:] = _pyaccel.tracking.findorbit4(acc,indices='open')
        M, T = _pyaccel.tracking.findm44(acc,closed_orbit=closed_orbit)

    A_InvB = lambda A,B: _np.linalg.solve(B.T, A.T).T
    InvA_B = lambda A,B: _np.linalg.solve(A, B)
    def get_C(DM_i, R0i, bpm, corr, length):
        # cxy --> orbit at bpm x due to kick in corrector y
        if bpm>corr:
            R_ij = A_InvB(R0i,T[corr]) # R0i/R0j
        else :
            R_ij = _np.dot(R0i, A_InvB(M, T[corr])) # Rij = R0i*M*inv(R0j)

        C = InvA_B(DM_i, R_ij)
        cxx = -(length/2)*C[0,0]   +   C[0,1]
        cyx = -(length/2)*C[2,0]   +   C[2,1]
        cxy = -(length/2)*C[0,2]   +   C[0,3]
        cyy = -(length/2)*C[2,2]   +   C[2,3]
        return cxx, cyx, cxy, cyy

    nr_hcms = len(hcms)
    nr_vcms = len(vcms)
    nr_bpms = len(bpms)
    mxx = _np.zeros((nr_bpms, nr_hcms))
    myx = _np.zeros((nr_bpms, nr_hcms))
    mxy = _np.zeros((nr_bpms, nr_vcms))
    myy = _np.zeros((nr_bpms, nr_vcms))
    len_hcms = [sum([acc[j].length for j in hcms[i]]) for i in range(len(hcms))]
    len_vcms = [sum([acc[j].length for j in vcms[i]]) for i in range(len(vcms))]
    D = _np.eye(M.shape[0])
    for i in range(nr_bpms):
        R_i = T[bpms[i]]
        DM_i = D - A_InvB(_np.dot(R_i,M), R_i) # I - R*M*inv(R)
        for j in range(nr_hcms):
            mxx[i,j], myx[i,j], *_ = get_C(DM_i, R_i, bpms[i],hcms[j][-1],len_hcms[j])
        for j in range(nr_vcms):
            *_, mxy[i,j], myy[i,j] = get_C(DM_i, R_i, bpms[i],vcms[j][-1],len_vcms[j])
    return mxx,mxy,myx,myy

def _get_kickangle(acc, indcs, plane):

    # correctors must behave as if the magnet was segmented:
    # kicks must have the same length as indices or be a scalar
    indcs, kicks = _pyaccel.lattice._process_args_errors(indcs,0.0)

    for ii in range(len(indcs)):
        meth = acc[indcs[ii][0]].pass_method
        if meth in {'corrector_pass'}:
            if plane in {'x','h'}:
                kicks[ii] = sum([acc[i].hkick for i in indcs[ii]])
            else:
                kicks[ii] = sum([acc[i].vkick for i in indcs[ii]])
        elif meth in {'str_mpole_symplectic4_pass','bnd_mpole_symplectic4_pass'}:
            if plane in {'x','h'}:
                kicks[ii] = sum([acc[i].hkick_polynom for i in indcs[ii]])
            else:
                kicks[ii] = sum([acc[i].vkick_polynom for i in indcs[ii]])
        else:
            raise TypeError('Element with pass_method "'+
                 '{0:s}" cannot be used as corrector.'.format(pass_method))
    return kicks

def _set_kickangle(acc, indcs, kicks, plane):

    # correctors must behave as if the magnet was segmented:
    # kicks must have the same length as indices or be a scalar
    indcs, kicks = _pyaccel.lattice._process_args_errors(indcs,kicks)

    for ii in range(len(indcs)):
        meth = acc[indcs[ii][0]].pass_method
        leng = len(indcs[ii])
        if meth in {'corrector_pass'}:
            if plane in {'x','h'}:
                for i in indcs[ii]: acc[i].hkick = kicks[ii]/leng
            else:
                for i in indcs[ii]: acc[i].vkick = kicks[ii]/leng
        elif meth in {'str_mpole_symplectic4_pass','bnd_mpole_symplectic4_pass'}:
            if plane in {'x','h'}:
                for i in indcs[ii]: acc[i].hkick_polynom = kicks[ii]/leng
            else:
                for i in indcs[ii]: acc[i].vkick_polynom = kicks[ii]/leng
        else:
            raise TypeError('Element with pass_method "'+
                 '{0:s}" cannot be used as corrector.'.format(pass_method))
