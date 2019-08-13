#!/usr/bin/env python-sirius

import numpy as np


'''Particle Swarm Optimization Algorithm for Minimization'''


class PSO:

    def __init__(self, nswarm=None):
        # Number of particles in the swarm # (Recommended is 10 + 2 * sqrt(d)) where d is the dimension of search space
        self._nswarm = nswarm
        # Factor of inertia of particles
        self._c_inertia = 0.7984
        # Factor of best position of individual particle
        self._c_indiv = 1.49618
        # Factor of best position ever reached by the swarm
        self._c_coll = self.c_indiv

        # Boundary limits of problem
        self._upper_limits = np.array([])
        self._lower_limits = np.array([])
        self.initialization()
        # The dimension is obtained by the definition of boundary limits
        self._ndim = len(self._upper_limits)
        self._check_initialization()
        # Elements of PSO
        self._position = np.array([])
        self._velocity = np.array([])
        self._best_particle = np.array([])
        self._best_global = np.array([])

    @property
    def c_inertia(self):
        return self._c_inertia

    @c_inertia.setter
    def c_inertia(self, value):
        self._c_inertia = value

    @property
    def c_indiv(self):
        return self._c_indiv

    @c_indiv.setter
    def c_indiv(self, value):
        self._c_indiv = value

    @property
    def c_coll(self):
        return self._c_coll

    @c_coll.setter
    def c_coll(self, value):
        self._c_coll = value

    @property
    def ndim(self):
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        self._ndim = value

    def _create_swarm(self):
        self._best_particle = np.zeros((self._nswarm, self._ndim))
        self._best_global = np.zeros(self._ndim)
        # Random initialization of swarm position inside the bounday limits
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._nswarm, self._ndim)
        self._position = dlim * rarray + self._lower_limits
        # The first individual contribution will be zero
        self._best_particle = self._position
        # Initializing with zero velocity
        self._velocity = np.zeros((self._nswarm, self._ndim))

    def _set_lim(self):
        # If particle position exceeds the boundary, set the boundary value
        for i in range(self._upper_limits.size):
            over = self._position[:, i] > self._upper_limits[i]
            under = self._position[:, i] < self._lower_limits[i]
            self._position[over, i] = self._upper_limits[i]
            self._position[under, i] = self._lower_limits[i]

    def initialization(self):
        pass

    def _check_initialization(self):
        if len(self._upper_limits) != len(self._lower_limits):
            print('Warning: Upper and Lower Limits has different lengths')

        if self._ndim != len(self._upper_limits):
            print('Warning: Dimension incompatible with limits!')

        if self._nswarm < int(10 + 2 * np.sqrt(self._ndim)):
            print('Warning: Swarm population lower than recommended!')

    def calc_merit_function(self):
        # Merit function must be a vector with the value for each particle
        return np.zeros(self._nswarm)

    def _update_position(self):
        r_indiv = self._c_indiv * np.random.rand()
        r_coll = self._c_coll * np.random.rand()
        # Inertial velocity
        self._velocity = self._c_inertia * self._velocity
        # Velocity dependent to distance from best individual position
        self._velocity += r_indiv * (self._best_particle - self._position)
        # Velocity dependent to distance from best global position
        self._velocity += r_coll * (self._best_global - self._position)
        # Update position and check boundary limits
        self._position = self._position + self._velocity
        self._set_lim()

    def _start_optimization(self, niter):
        self._create_swarm()

        f_old = np.zeros(self._nswarm)
        f_new = np.zeros(self._nswarm)

        # History of best position and merit function over iteractions
        best_pos_hstry = np.zeros([niter, self._ndim])
        best_fig_hstry = np.zeros(niter)

        f_old = self.calc_merit_function()
        self._best_global = self._best_particle[np.argmin(f_old), :]

        k = 0
        while k < niter:
            print('>>> Iteraction Number:' + str(k+1))
            self._update_position()
            f_new = self.calc_merit_function()
            improve = f_new < f_old
            # Update best individual position and merit function for comparison only if the merit function is lower
            self._best_particle[improve, :] = self._position[improve, :]
            f_old[improve] = f_new[improve]
            self._best_global = self._best_particle[np.argmin(f_old), :]
            if improve.any():
                print('Global best updated:' + str(self._best_global))
                print('Figure of merit updated:' + str(np.min(f_old)))
            # Storing history of best global position and best figure of merit
            best_pos_hstry[k, :] = self._best_global
            best_fig_hstry[k] = np.min(f_old)
            with open('pos_swarm.txt', 'a') as f_pos:
                f_pos.write('Step ' + str(k+1) + ' \n')
                np.savetxt(f_pos, self._position, fmt='%+.8e')
            k += 1

        print('Best Position Found:' + str(self._best_global))
        print('Best Figure of Merit Found:' + str(np.min(f_old)))
        np.savetxt('best_pos_history.txt', best_pos_hstry)
        np.savetxt('best_fig_history.txt', best_fig_hstry)
        return best_pos_hstry, best_fig_hstry


''' Simulated Annealing Algorithm for Minimization'''


class SimulAnneal:

    def __init__(self):
        # Boundary Limits
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
        self.initialization()
        # Dimension of search space is obtained by boundary limits
        self._ndim = len(self._lower_limits)
        self._check_initialization()

    def initialization(self):
        pass

    def _check_initialization(self):
        pass

    def _set_lim(self):
        # If particle position exceeds the boundary, set the boundary value
        over = self._position > self._upper_limits
        under = self._position < self._lower_limits
        self._position[over] = self._upper_limits[over]
        self._position[under] = self._lower_limits[under]

    def calc_merit_function(self):
        return 0

    def _init_pos(self):
        # Random initialization position inside the bounday limits
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._ndim)
        self._position = dlim * rarray + self._lower_limits

    def _random_change(self):
        # Random change applied in the current position
        dlim = self._max_delta
        rarray = np.random.rand(self._ndim)
        self._delta = dlim * rarray
        self._position = self._position + self._delta
        self._set_lim()

    def _start_optimization(self, niter):
        self._init_pos()

        f_old = self.calc_merit_function()
        best = self._position
        # Number of accepted solutions
        n_acc = 0
        # Number of iteraction without accepting solutions
        nu = 0

        for k in range(niter):
            # Flag that a solution was accepted
            flag_acc = False
            self._random_change()
            f_new = self.calc_merit_function()

            if f_new < f_old:
                # Accepting solution if it reduces the merit function
                flag_acc = True
                best = self._position
                print('Better solution found! ' + str(best))
                nu = 0
            elif f_new > f_old and self._temperature != 0:
                # If solution increases the merit function there is a chance to accept it
                df = f_new - f_old
                if np.random.rand() < np.exp(- df / self._temperature):
                    flag_acc = True
                    print('Worse solution accepted! ' + str(self._position))
                    print('Temperature is: ' + str(self._temperature))
                else:
                    flag_acc = False
            else:
                # If temperature is zero the algorithm only accepts good solutions
                flag_acc = False

            if flag_acc:
                # Stores the number of accepted solutions
                f_old = f_new
                n_acc += 1
                print('Number of accepted solutions: ' + str(n_acc))
            else:
                self._position = self._position - self._delta
                nu += 1

            if self._temperature != 0:
                # Reduces the temperature based on number of iteractions without accepting solutions
                # Ref: An Optimal Cooling Schedule Using a Simulated Annealing Based Approach - A. Peprah, S. Appiah, S. Amponsah
                phi = 1 / (1 + 1 / np.sqrt((k+1) * (nu + 1) + nu))
                self._temperature = phi * self._temperature

        print('Best solution is: ' + str(best))
        print('Best figure of merit is: ' + str(f_old))
        print('Number of accepted solutions: ' + str(n_acc))


'''Multidimensional Simple Scan method for Minimization'''


class SimpleScan:

    def __init__(self):
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._position = np.array([])
        self._delta = np.array([])
        self._curr_dim = 0
        self.initialization()
        self._ndim = len(self._upper_limits)
        self._check_initialization()

    def initialization(self):
        pass

    def _check_initialization(self):
        pass

    def calc_merit_function(self):
        return np.zeros(self._ndim), np.zeros(self._ndim)

    def _start_optimization(self, npoints):
        self._delta = np.zeros(npoints)
        f = np.zeros(self._ndim)
        best = np.zeros(self._ndim)

        for i in range(self._ndim):
            self._delta = np.linspace(
                                self._lower_limits[i],
                                self._upper_limits[i],
                                npoints)
            self._curr_dim = i
            f[i], best[i] = self.calc_merit_function()
            self._position[i] = best[i]

        print('Best result is: ' + str(best))
        print('Figure of merit is: ' + str(np.min(f)))


class GA:

    def __init__(self, npop, nparents, mutrate=0.01):
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
        pass

    def _check_initialization(self):
        pass

    def calc_merit_function(self):
        return np.zeros(self._npop)

    def _create_pop(self):
        # Random initialization of elements inside the bounday limits
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._npop, self._ndim)
        self._indiv = dlim * rarray + self._lower_limits

    def _select_parents(self, f):
        # Select parents based on best ranked ones
        ind_sort = np.argsort(f)
        return self._indiv[ind_sort[:self._nparents], :]

    def _crossover(self, parents):
        child = np.zeros([self._nchildren, self._ndim])
        # Create list of random pairs to produce children
        par_rand = np.random.randint(0, self._nparents, [self._nchildren, 2])
        # Check if occurs that two parents are the same
        equal_par = par_rand[:, 0] == par_rand[:, 1]

        while equal_par.any():
            # While there is two parents that are the same, randomly choose another first parent
            par_rand[equal_par, 0] = np.random.randint(
                0, self._nparents, np.sum(equal_par))
            equal_par = par_rand[:, 0] == par_rand[:, 1]

        for i in range(self._nchildren):
            for j in range(self._ndim):
                # For each child and for each gene, choose which gene will be inherited from parent 1 or parent 2 (each parent has 50% of chance)
                if np.random.rand(1) < 0.5:
                    child[i, j] = parents[par_rand[i, 0], j]
                else:
                    child[i, j] = parents[par_rand[i, 1], j]
        return child

    def _mutation(self, child):
        for i in range(self._nchildren):
            # For each child, with MutRate of chance a mutation can occur
            if np.random.rand(1) < self._mutrate:
                # Choose the number of genes to perform mutation (min is 1 and max is the maximum number of genes)
                num_mut = np.random.randint(1, self._ndim)
                # Choose which genes are going to be changed
                gen_mut = np.random.randint(0, self._ndim, num_mut)
                # Mutation occurs as a new random initialization
                dlim = (self._upper_limits - self._lower_limits)[gen_mut]
                rarray = np.random.rand(num_mut)
                change = dlim * rarray + self._lower_limits[gen_mut]
                child[i, gen_mut] = change
        return child

    def _start_optimization(self, niter):
        self._create_pop()

        for k in range(niter):
            print('Generation number ' + str(k+1))
            fout = self.calc_merit_function()
            print('Best Figure of Merit: ' + str(np.min(fout)))
            print(
                'Best Configuration: ' + str(self._indiv[np.argmin(fout), :]))
            parents = self._select_parents(fout)
            children = self._crossover(parents)
            children_mut = self._mutation(children)
            self._indiv[:self._nparents, :] = parents
            self._indiv[self._nparents:, :] = children_mut


''' Powell Conjugated Direction Search Method for Minimization


class Powell():

    GOLDEN = (np.sqrt(5) - 1)/2

    def __init__(self):
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._position = np.array([])
        self._delta = np.array([])
        self.initialization()
        self._ndim = len(self._upper_limits)
        self._check_initialization()

    def initialization(self):
        pass

    def _check_initialization(self):
        pass



    def calc_merit_function(self):
        return np.zeros(self._ndim)

    def golden_search(self):
        k = 0
        x = self._position
        x_upper = x[1]
        x_lower = x[0]
        d = GOLDEN * (x_upper - x_lower)
        x1 = x_lower + d
        x2 = x_upper - d
        f1 = self.calc_merit_func(x1)
        f2 = self.calc_merit_func(x2)

        while k < self._nint:
            if f1 > f2:
                x_lower = x2
                x2 = x1
                f2 = f1
                x1 = x_lower + GOLDEN * (x_upper - x_lower)
                f1 = self.calc_merit_func(x1)
            elif f2 > f1:
                x_upper = x1
                x1 = x2
                f1 = f2
                x2 = x_upper - GOLDEN * (x_upper - x_lower)
                f2 = self.calc_merit_func(x2)
            k += 1
        return x_lower, x_upper

    def line_scan(self):
        pass

    def _start_optimization(self, niter):
        pass
        '''
