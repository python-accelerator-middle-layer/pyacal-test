#!/usr/bin/env python-sirius

import numpy as np


'''Particle Swarm Optimization Algorithm for Minimization'''


class PSO:
    """."""

    def __init__(self, save=False):
        """."""
        # Number of particles in the swarm # (Recommended is 10 + 2 * sqrt(d))
        # where d is the dimension of search space
        self._nswarm = []
        self._niter = []
        self._coeff_inertia = 0.7984  # Inertia
        self._coeff_indiv = 1.49618  # Best position of individual particle
        self._coeff_coll = self._coeff_indiv  # Best position ever reached by
        # the swarm

        # Boundary limits of problem
        self._upper_limits = np.array([])
        self._lower_limits = np.array([])
        self.initialization()
        self._check_initialization()
        # Elements of PSO
        self._position = np.array([])
        self._velocity = np.array([])
        self._best_indiv = np.array([])
        self._best_global = np.array([])
        self.f_init = []

        self._flag_save = save

    @property
    def coeff_inertia(self):
        """."""
        return self._coeff_inertia

    @coeff_inertia.setter
    def coeff_inertia(self, value):
        """."""
        self._coeff_inertia = value

    @property
    def coeff_indiv(self):
        """."""
        return self._coeff_indiv

    @coeff_indiv.setter
    def coeff_indiv(self, value):
        """."""
        self._coeff_indiv = value

    @property
    def coeff_coll(self):
        """."""
        return self._coeff_coll

    @coeff_coll.setter
    def coeff_coll(self, value):
        """."""
        self._coeff_coll = value

    @property
    def ndim(self):
        """."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        """."""
        self._ndim = value

    @property
    def nswarm(self):
        """."""
        return self._nswarm

    @nswarm.setter
    def nswarm(self, value):
        """."""
        self._nswarm = value

    @property
    def niter(self):
        """."""
        return self._niter

    @niter.setter
    def niter(self, value):
        """."""
        self._niter = value

    @property
    def position(self):
        """."""
        return self._position

    @position.setter
    def position(self, value):
        """."""
        self._position = value

    def initialization(self):
        """."""
        raise NotImplementedError

    def _check_initialization(self):
        """."""
        if len(self._upper_limits) != len(self._lower_limits):
            raise Exception(
                'Upper and Lower Limits has different lengths')

        if self._ndim != len(self._upper_limits):
            raise Exception(
                'Dimension incompatible with limits!')

        if self._nswarm < int(10 + 2 * np.sqrt(self._ndim)):
            raise Warning(
                'Swarm population lower than recommended!')

    def _create_swarm(self):
        """."""
        self._best_indiv = np.zeros((self._nswarm, self._ndim))
        self._best_global = np.zeros(self._ndim)
        # Random initialization of swarm position inside the bounday limits
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._nswarm, self._ndim)
        self._position = dlim * rarray + self._lower_limits
        # The first individual contribution will be zero
        self._best_indiv = self._position
        # Initializing with zero velocity
        self._velocity = np.zeros((self._nswarm, self._ndim))

    def init_obj_func(self):
        """."""
        raise NotImplementedError

    def _update_position(self):
        """."""
        r_indiv = self._coeff_indiv * np.random.rand()
        r_coll = self._coeff_coll * np.random.rand()
        # Inertial velocity
        self._velocity = self._coeff_inertia * self._velocity
        # Velocity dependent to distance from best individual position
        self._velocity += r_indiv * (self._best_indiv - self._position)
        # Velocity dependent to distance from best global position
        self._velocity += r_coll * (self._best_global - self._position)
        # Update position and check boundary limits
        self._position = self._position + self._velocity
        self._check_lim()

    def _check_lim(self):
        """."""
        # If particle position exceeds the boundary, set the boundary value
        for i in range(self._upper_limits.size):
            over = self._position[:, i] > self._upper_limits[i]
            under = self._position[:, i] < self._lower_limits[i]
            self._position[over, i] = self._upper_limits[i]
            self._position[under, i] = self._lower_limits[i]

    def set_limits(self, upper=None, lower=None):
        """."""
        self._upper_limits = upper
        self._lower_limits = lower
        self.ndim = len(upper)
        if not self.nswarm:
            self.nswarm = int(10 + 2 * np.sqrt(self.ndim))

    def get_change(self):
        """."""
        raise NotImplementedError

    def set_change(self):
        """."""
        raise NotImplementedError

    def _save_data(self, k, f, fbest):
        """."""
        with open('pos_PSO.txt', 'a') as f_pos:
            if k == 0:
                f_pos.write('NEW RUN'.center(50, '='))
            f_pos.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_pos, self._position, fmt='%+.8e')
        with open('fig_PSO.txt', 'a') as f_fig:
            if k == 0:
                f_fig.write('NEW RUN'.center(50, '='))
            f_fig.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_fig, f, fmt='%+.8e')
        with open('best_pos_history_PSO.txt', 'a') as f_posh:
            if k == 0:
                f_posh.write('NEW RUN'.center(50, '='))
            f_posh.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_posh, self._best_global, fmt='%+.8e')
        with open('best_fig_history_PSO.txt', 'a') as f_figh:
            if k == 0:
                f_figh.write('NEW RUN'.center(50, '='))
            f_figh.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_figh, np.array([fbest]), fmt='%+.8e')

    def calc_obj_fun(self):
        """Return a vector for every particle evaluation."""
        raise NotImplementedError

    def start_optimization(self):
        """."""
        self._create_swarm()

        f_old = np.zeros(self._nswarm)
        f_new = np.zeros(self._nswarm)

        # History of best position and merit function over iteractions
        best_pos_hstry = np.zeros([self.niter, self._ndim])
        best_fig_hstry = np.zeros(self.niter)

        print('>>> Iteraction Number:1')
        f_old = self.calc_obj_fun()
        if np.min(f_old) < self.f_init:
            self._best_global = self._best_indiv[np.argmin(f_old), :]
        else:
            min_idx = np.argmin(f_old)
            self._position[min_idx, :] = np.zeros([1, self._ndim])
            self._best_indiv[min_idx, :] = self._position[min_idx, :]
            self._best_global[min_idx, :] = self._best_indiv[min_idx, :]

        best_pos_hstry[0, :] = self._best_global
        best_fig_hstry[0] = np.min(f_old)
        if self._flag_save:
            self._save_data(k=0, f=f_old, fbest=best_fig_hstry[0])
        print('Best particle: ' + str(np.argmin(f_old)+1))
        print('Obj. Func.:' + str(np.min(f_old)))

        k = 1
        while k < self.niter:
            print('>>> Iteraction Number:' + str(k+1))
            self._update_position()
            f_new = self.calc_obj_fun()
            improve = f_new < f_old
            if improve.any():
                # Update best individual position and merit function for
                # comparison only if the merit function is lower
                self._best_indiv[improve, :] = self._position[improve, :]
                if np.min(f_new) < np.min(f_old):
                    self._best_global = self._best_indiv[
                        np.argmin(f_new), :]
                    print('Update global best!')
                    print(
                        'Best particle: ' + str(np.argmin(f_new)+1))
                    print('Obj. Func.:' + str(np.min(f_new)))
                    f_old[improve] = f_new[improve]
                else:
                    print('Best particle: ' + str(np.argmin(f_new)+1))
                    print('Obj. Func.:' + str(np.min(f_new)))

            best_pos_hstry[k, :] = self._best_global
            best_fig_hstry[k] = np.min(f_old)
            if self._flag_save:
                self._save_data(k=k, f=f_new, fbest=best_fig_hstry[k])
            k += 1

        print('Best Position Found:' + str(self._best_global))
        print('Best Obj. Func. Found:' + str(np.min(f_old)))
        # np.savetxt('best_pos_history.txt', best_pos_hstry)
        # np.savetxt('best_fig_history.txt', best_fig_hstry)
        return best_pos_hstry, best_fig_hstry


''' Simulated Annealing Algorithm for Minimization'''


class SimulAnneal:
    """."""

    @property
    def ndim(self):
        """."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        """."""
        self._ndim = value

    @property
    def position(self):
        """."""
        return self._position

    @position.setter
    def position(self, value):
        """."""
        self._position = value

    @property
    def niter(self):
        """."""
        return self._niter

    @niter.setter
    def niter(self, value):
        """."""
        self._niter = value

    @property
    def temperature(self):
        """."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """."""
        self._temperature = value

    def __init__(self, save=False):
        """."""
        # Boundary Limits
        self._ndim = []
        self._niter = []
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        # Maximum variation to be applied
        self._max_delta = np.array([])
        # Reference configuration
        self._position = np.array([])
        # Variation to be applied
        self._delta = np.array([])
        # Initial temperature of annealing
        self._temperature = 0
        self._flag_save = save
        self.f_init = []
        self.initialization()

    def initialization(self):
        """."""
        raise NotImplementedError

    def _check_lim(self):
        # If particle position exceeds the boundary, set the boundary value
        over = self._position > self._upper_limits
        under = self._position < self._lower_limits
        self._position[over] = self._upper_limits[over]
        self._position[under] = self._lower_limits[under]

    def set_limits(self, upper=None, lower=None):
        """."""
        self._upper_limits = upper
        self._lower_limits = lower
        self.ndim = len(upper)

    def set_deltas(self, dmax=None):
        """."""
        self.ndim = len(dmax)
        self._max_delta = dmax

    def get_change(self):
        """."""
        raise NotImplementedError

    def set_change(self):
        """."""
        raise NotImplementedError

    def calc_obj_fun(self):
        """Return a number."""
        raise NotImplementedError

    def _random_change(self):
        # Random change applied in the current position
        dlim = self._max_delta
        rarray = 2 * np.random.rand(self.ndim) - 1  # [-1,1]
        self._delta = dlim * rarray
        self._position = self._position + self._delta
        # self._check_lim()

    def _save_data(self, k, f, acc=False, nacc=None, bp=None, bf=None):
        """."""
        with open('pos_SA.txt', 'a') as f_pos:
            if k == 0:
                f_pos.write('NEW RUN'.center(50, '='))
            f_pos.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_pos, self._position, fmt='%+.8e')
        with open('fig_SA.txt', 'a') as f_fig:
            if k == 0:
                f_fig.write('NEW RUN'.center(50, '='))
            f_fig.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_fig, np.array([f]), fmt='%+.8e')
        if acc:
            with open('best_pos_history_SA.txt', 'a') as f_posh:
                if nacc == 1:
                    f_posh.write('NEW RUN'.center(50, '='))
                f_posh.write('Accep. Solution ' + str(nacc+1) + ' \n')
                np.savetxt(f_posh, bp[nacc, :], fmt='%+.8e')
            with open('best_fig_history_SA.txt', 'a') as f_figh:
                if nacc == 1:
                    f_figh.write('NEW RUN'.center(50, '='))
                f_figh.write('Accep. Solution ' + str(nacc+1) + ' \n')
                np.savetxt(f_figh, np.array([bf[nacc]]), fmt='%+.8e')

    def start_optimization(self):
        """."""
        bpos_hstry = np.zeros([self.niter, self.ndim])
        bfig_hstry = np.zeros([self.niter])

        f_old = self.calc_obj_fun()

        if self.f_init < f_old:
            f_old = self.f_init
            self._position = np.zeros([1, self.ndim])

        bfig_hstry[0] = f_old
        bpos_hstry[0, :] = self._position
        # Number of accepted solutions
        n_acc = 0
        # Number of iteraction without accepting solutions
        nu = 0

        if self._flag_save:
            self._save_data(k=0, f=f_old, acc=False)

        for k in range(self.niter):
            # Flag that a solution was accepted
            flag_acc = False
            self._random_change()
            print('>>> Iteraction Number:' + str(k+1))
            f_new = self.calc_obj_fun()

            if f_new < f_old:
                # Accepting solution if it reduces the merit function
                flag_acc = True
                nu = 0
            elif f_new > f_old and self._temperature != 0:
                # If solution increases the merit function there is a chance
                # to accept it
                df = f_new - f_old
                if np.random.rand() < np.exp(- df / self._temperature):
                    flag_acc = True
                    print('Worse solution accepted! ' + str(self._position))
                    print('Temperature is: ' + str(self._temperature))
                else:
                    flag_acc = False
            else:
                # If temperature is zero the algorithm only accepts good
                # solutions
                flag_acc = False

            if flag_acc:
                # Stores the number of accepted solutions
                f_old = f_new
                n_acc += 1
                bpos_hstry[n_acc, :] = self._position
                bfig_hstry[n_acc] = f_old
                print('Better solution found! Obj. Func: {:5f}'.format(f_old))
                print('Number of accepted solutions: ' + str(n_acc))
            else:
                self._position = self._position - self._delta
                nu += 1

            if self._flag_save:
                self._save_data(
                    k=k+1, f=f_old, acc=flag_acc, nacc=n_acc, bp=bpos_hstry,
                    bf=bfig_hstry)

            if self._temperature != 0:
                # Reduces the temperature based on number of iteractions
                # without accepting solutions
                # Ref: An Optimal Cooling Schedule Using a Simulated Annealing
                # Based Approach - A. Peprah, S. Appiah, S. Amponsah
                phi = 1 / (1 + 1 / np.sqrt((k+1) * (nu + 1) + nu))
                self._temperature = phi * self._temperature
        if n_acc:
            bpos_hstry = bpos_hstry[:n_acc, :]
            bfig_hstry = bfig_hstry[:n_acc]

            print('Best solution found: ' + str(bpos_hstry[-1, :]))
            print(
                'Best Obj. Func. found: ' + str(bfig_hstry[-1]))
            print('Number of accepted solutions: ' + str(n_acc))
        else:
            bpos_hstry = bpos_hstry[0, :]
            bfig_hstry = bfig_hstry[0]
            print('It was not possible to find a better solution...')

        return bpos_hstry, bfig_hstry


'''Multidimensional Simple Scan method for Minimization'''


class SimpleScan:

    def __init__(self):
        """."""
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._position = np.array([])
        self._delta = np.array([])
        self._curr_dim = 0
        self.initialization()
        self._ndim = len(self._upper_limits)

    def initialization(self):
        """."""
        raise NotImplementedError

    def calc_obj_fun(self):
        """Return arrays with dimension of search space."""
        raise NotImplementedError

    def start_optimization(self, npoints):
        """."""
        self._delta = np.zeros(npoints)
        f = np.zeros(self._ndim)
        best = np.zeros(self._ndim)

        for i in range(self._ndim):
            self._delta = np.linspace(
                                self._lower_limits[i],
                                self._upper_limits[i],
                                npoints)
            self._curr_dim = i
            f[i], best[i] = self.calc_obj_fun()
            self._position[i] = best[i]

        print('Best result is: ' + str(best))
        print('Figure of merit is: ' + str(np.min(f)))


class GA:
    """."""

    def __init__(self, npop, nparents, mutrate=0.01):
        """."""
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._indiv = np.array([])
        self.initialization()
        # Dimension of search space is obtained by boundary limits
        self._ndim = len(self._upper_limits)
        # Population size
        self._npop = npop
        # Number of parents to be selected
        self._nparents = nparents
        # Number of offspring generated from parents
        self._nchildren = self._npop - self._nparents
        # Mutation rate (Default = 1%)
        self._mutrate = mutrate

    def initialization(self):
        """."""
        raise NotImplementedError

    def calc_obj_fun(self):
        """Return array with size equal to the population size."""
        raise NotImplementedError

    def _create_pop(self):
        """."""
        # Random initialization of elements inside the bounday limits
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._npop, self._ndim)
        self._indiv = dlim * rarray + self._lower_limits

    def _select_parents(self, f):
        """."""
        # Select parents based on best ranked ones
        ind_sort = np.argsort(f)
        return self._indiv[ind_sort[:self._nparents], :]

    def _crossover(self, parents):
        """."""
        child = np.zeros([self._nchildren, self._ndim])
        # Create list of random pairs to produce children
        par_rand = np.random.randint(0, self._nparents, [self._nchildren, 2])
        # Check if occurs that two parents are the same
        equal_par = par_rand[:, 0] == par_rand[:, 1]

        while equal_par.any():
            # While there is two parents that are the same, randomly choose
            # another first parent
            par_rand[equal_par, 0] = np.random.randint(
                0, self._nparents, np.sum(equal_par))
            equal_par = par_rand[:, 0] == par_rand[:, 1]

        for i in range(self._nchildren):
            for j in range(self._ndim):
                # For each child and for each gene, choose which gene will be
                # inherited from parent 1 or parent 2 (each parent has 50% of
                # chance)
                if np.random.rand(1) < 0.5:
                    child[i, j] = parents[par_rand[i, 0], j]
                else:
                    child[i, j] = parents[par_rand[i, 1], j]
        return child

    def _mutation(self, child):
        """."""
        for i in range(self._nchildren):
            # For each child, with MutRate of chance a mutation can occur
            if np.random.rand(1) < self._mutrate:
                # Choose the number of genes to perform mutation (min is 1 and
                # max is the maximum number of genes)
                num_mut = np.random.randint(1, self._ndim)
                # Choose which genes are going to be changed
                gen_mut = np.random.randint(0, self._ndim, num_mut)
                # Mutation occurs as a new random initialization
                dlim = (self._upper_limits - self._lower_limits)[gen_mut]
                rarray = np.random.rand(num_mut)
                change = dlim * rarray + self._lower_limits[gen_mut]
                child[i, gen_mut] = change
        return child

    def start_optimization(self, niter):
        """."""
        self._create_pop()

        for k in range(niter):
            print('Generation number ' + str(k+1))
            fout = self.calc_obj_fun()
            print('Best Figure of Merit: ' + str(np.min(fout)))
            print(
                'Best Configuration: ' + str(self._indiv[np.argmin(fout), :]))
            parents = self._select_parents(fout)
            children = self._crossover(parents)
            children_mut = self._mutation(children)
            self._indiv[:self._nparents, :] = parents
            self._indiv[self._nparents:, :] = children_mut
