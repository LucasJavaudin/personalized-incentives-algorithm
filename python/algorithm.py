#############
#  Imports  #
#############

import time
import os
import itertools
import numpy as np
import progressbar
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit

# Define colors for the graphs.
COLOR_1 = '#608f42'
COLOR_2 = '#90ced6'
COLOR_3 = '#54251e'
COLOR = [COLOR_1, COLOR_2, COLOR_3]

# Update matplotlib parameters.
params = {'text.usetex': True,
          'figure.dpi': 200,
          'font.size': 10,
          'font.serif': [],
          'font.sans-serif': [],
          'font.monospace': [],
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'axes.linewidth': .6,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'font.family': 'serif'}
plt.rcParams.update(params)


###########
#  Class  #
###########

class Data:

    """A Data object stores the information on the
    alternatives of the individuals.

    A Data object has a list attribute with I elements where I is the number of
    individuals.
    Each element is a J_i x 2 numpy array where J_i is the number of
    alternatives of individual i.
    The attribute individuals is an indicator of the number of individuals in
    the data.
    The attribute alternatives_per_individual indicates the number of
     alternatives for all individuals.
    The attribute total_alternatives indicates the total number of
    alternatives.
    The attribute generated_data is True if the data are generated.
    The attribute is_sorted is True if the data are sorted by utility and then
    by energy consumption.
    The attribute pareto_dominated_removed is True if the Pareto-dominated
     alternatives are removed from the data.
    The attribute efficiency_dominated_removed is True if the
    efficiency-dominated alterantives are removed from the data.
    """

    def __init__(self):
        """Initialize a Data object.

        The value of list is an empty list.
        The value of individuals is 0.
        The value of generated_data is False by default.
        The data is uncleaned and unsorted by default.
        """
        self.list = list()
        self.alternative_labels = list()
        self.individuals = 0
        self.alternatives_per_individual = list()
        self.total_alternatives = 0
        self.generated_data = False
        self.is_sorted = False
        self.pareto_dominated_removed = False
        self.nb_pareto_removed = 0
        self.efficiency_dominated_removed = False
        self.read_time = None
        self.generating_time = None
        self.output_data_time = None
        self.output_characteristics_time = None
        self.sorting_time = None
        self.pareto_removing_time = None
        self.efficiency_removing_time = None

    def _append(self, narray):
        """Append the specified numpy array to the Data object.

        Increase the number of individuals by 1 and update the number of
        alternatives.
        The numpy array should have 2 columns.
        """
        assert isinstance(narray, np.ndarray), \
            'Tried to append a non np.array object to a Data object'
        assert narray.shape[1] == 2, \
            'Tried to append a numpy array which does not have 2 columns'
        self.list.append(narray)
        # The number of individuals increases by 1.
        self.individuals += 1
        # Count the number of alternatives in the array.
        n = narray.shape[0]
        # Add the number of alternatives for the individual.
        self.alternatives_per_individual.append(n)
        # Update the total number of alternatives.
        self.total_alternatives += n
        # The data are a priori no longer sorted and cleaned.
        self.is_sorted = False
        self.pareto_dominated_removed = False

    def read(self, filename, delimiter=',', comment='#', verbose=True):
        """Read data from an input file.

        Lines can be commented with the specified character.
        There are twice as many uncommented lines as individuals.
        The odd line contains the utility of the alternatives, separated by the
        specified delimiter.
        The even line contains the energy consumption of the alternatives,
        separated by the specified delimiter.

        :filename: string with the name of the file containing the data
        :delimiter: the character used to separated the utility and the energy
        consumption of the alternatives, default is comma
        :comment: line starting with this string are not read, should be a
        string, default is #
        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        # Try to open file filename and return FileNotFoundError if python is
        # unable to open the file.
        try:
            input_file = open(filename, 'r')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if verbose:
            # Print a progress bar of unknown duration.
            bar, counter = _unknown_custom_bar(
                    'Importing data from file',
                    'lines imported')
        # The variable odd is True when the line number is odd (utility) and
        # false when the line number is even (energy consumption).
        odd = True
        for line in input_file:
            # The line starting by comment are not read.
            if not line.startswith(comment):
                # For odd lines, the values are stored in the numpy array
                # utility.
                if odd:
                    utility = np.fromstring(line, sep=delimiter)
                    odd = False
                # For even lines, the values are stored in the numpy array
                # energy.
                else:
                    energy = np.fromstring(line, sep=delimiter)
                    # Make sure that utility and energy are of the same size.
                    assert utility.shape == energy.shape, \
                        """In the input file, each individuals should have two
                        lines with the same number of values"""
                    # Append the numpy array (utility, energy) to the data
                    # object.
                    line = np.stack((utility, energy), axis=1)
                    self._append(line)
                    self.alternative_labels.append(np.arange(len(utility)))
                    odd = True
                    if verbose:
                        # Update the progress bar and increase the counter by
                        # 1.
                        bar.update(counter)
                        counter += 1
        input_file.close()
        if verbose:
            bar.finish()
        # Store the time spent to read data.
        self.read_time = time.time() - init_time

    def read_dataframe(self, df, indiv_col, utility_col, energy_col,
                       label_col=None, verbose=True):
        """Read data from a pandas DataFrame.

        :df: pandas DataFrame containing all the data
        :indiv_col: string indicating the column with the id of the individuals
        :utility_col: string indicating the column with the individual
        utilities for each alternative
        :energy_col: string indicating the column with the energy for each
        alternative
        :label_col: string indicating the column with the label of each
        alternative
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        # Retrieve the ids of the individuals.
        indiv_ids = df[indiv_col].unique()
        alt_per_indiv = list(df.groupby(indiv_col).size())
        if verbose:
            # Print a progress bar.
            bar = _known_custom_bar(
                np.sum(alt_per_indiv),
                'Importing data from dataframe',
            )
        values = df[[utility_col, energy_col]].values
        if label_col is not None:
            labels = df[label_col].values
        i = 0
        for j in alt_per_indiv:
            self.list.append(values[i:i+j])
            if label_col is not None:
                self.alternative_labels.append(labels[i:i+j])
            else:
                self.alternative_labels.append(np.arange(j))
            i += j
            if verbose:
                # Update the progress bar and increase the counter by
                # 1.
                bar.update(i)
        self.individuals += len(indiv_ids)
        self.alternatives_per_individual.extend(alt_per_indiv)
        self.total_alternatives += np.sum(alt_per_indiv)
        self.is_sorted = False
        self.pareto_dominated_removed = False
        if verbose:
            bar.finish()
        # Store the time spent to read data.
        self.read_time = time.time() - init_time

    def old_generate(self, individuals=1000, mean_nb_alternatives=10,
                     utility_mean=1, utility_sd=1, random_utility_parameter=10,
                     alpha=1, beta=0, gamma=1, use_poisson=True,
                     use_gumbel=False, verbose=True):
        """Generate a random data (old method).

        First, the number of alternatives of each individual is generated from
        a Poisson law or is set deterministically.
        For each alternative, utility and energy consumption are randomly
        generated.
        Utility is decomposed into a deterministic part and a random part.
        The deterministic part is drawn from a log-normal distribution.
        The random part is drawn from a Gumbel distribution or a Logistic
        distribution.
        Energy consumption is a function of the deterministic utility:
        Energy = alpha * deterministic_utility^gamma + beta.

        :individuals: number of individuals in the generated data, should be an
        integer greater than 2, default is 1000
        :mean_nb_alternatives: the parameter of the Poisson law used to
        generate the number of alternatives is (mean_nb_alternatives - 1),
        should be strictly greater than 1, should be an integer if use_poisson
        is false, default is 10
        :utility_mean: mean parameter when generating the utility of the
        alternatives, default is 1
        :utility_sd: standard-deviation parameter when generating the utility
        of the alternatives, should be positive, default is 1
        :random_utility_parameter: parameter used when generating the random
        part of the utility, should be positive, default is 10
        :alpha: parameter used to represent a multiplicative relation between
        utility and energy consumption, default is 1
        :beta: parameter used to represent an additive relation between utility
        and energy consumption, default is 0
        :gamma: parameter used to represent an exponential relation between
        utility and energy consumption, default is 1
        :use_poisson: if True the number of alternatives is drawn from a
        modified Poisson distribution with parameter
        (mean_nb_alternatives - 1), else the number of alternatives is equal to
        mean_nb_alternatives for all individuals, default is True
        :use_gumbel: if True the Gumbel distribution is used to generate the
        random utility, else the Logistic distribution is used, default is
        False
        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert individuals > 1 and isinstance(individuals, (int, np.int64)), \
            'The parameter individuals should be an integer greater than 2'
        assert mean_nb_alternatives > 1, \
            """The parameter mean_nb_alternatives should be stricty greater
            than 1"""
        assert isinstance(mean_nb_alternatives, int) or use_poisson, \
            """The parameter mean_nb_alternatives should be an integer when
            use_poisson is false"""
        assert utility_sd >= 0, \
            'The parameter utility_sd should be positive'
        assert random_utility_parameter >= 0, \
            'The parameter random_utility_parameter should be positive'
        if verbose:
            # Print a progress bar of duration individuals.
            bar = _known_custom_bar(individuals, 'Generating data')
        # The Data object is generated from a random sample
        self.generate_data = True
        if use_poisson:
            # Generate a list with the number of alternatives of all
            # individuals from a Poisson law.
            nb_alternatives = np.random.poisson(mean_nb_alternatives-1,
                                                individuals) + 1
        else:
            # All individuals have the same number of alternatives.
            nb_alternatives = np.repeat(mean_nb_alternatives, individuals)
        for i in range(individuals):
            # Generate deterministic utility from a log-normal distribution
            # with mean utility_mean and standard-deviation utility_sd.
            # The number of values generated is equal to the number of
            # alternatives of the individual i
            deterministic_utility = np.random.lognormal(
                    utility_mean, utility_sd, size=nb_alternatives[i]
            )
            # Generate random utility from a Gumbel distribution is use_gumbel
            # is True or from a Logistic distribution if use_gumbel is False.
            # The parameter of the random term is random_utility_parameter.
            if use_gumbel:
                random_utility = np.random.gumbel(0, random_utility_parameter,
                                                  size=nb_alternatives[i])
            else:
                random_utility = np.random.logistic(
                        0, random_utility_parameter, size=nb_alternatives[i]
                )
            # Total utility is the sum of the deterministic utility and the
            # random utility.
            utility = deterministic_utility + random_utility
            # The energy consumption is a function of the deterministic utility
            # with parameters alpha, beta and gamma.
            energy = alpha * np.power(deterministic_utility, gamma) + beta
            # Append the numpy array (utility, energy) to the data object.
            individual = np.stack((utility, energy), axis=1)
            self._append(individual)
            if verbose:
                # Update the progress bar.
                bar.update(i)
        if verbose:
            bar.finish()
        # Inform that the data were randomly generated.
        self.generated_data = True
        # Store the time spent to generate data.
        self.generating_time = time.time() - init_time

    def generate(self, individuals=1000, mean_nb_alternatives=10,
                 utility_variance=1, energy_variance=1, correlation=0,
                 use_poisson=True, seed=None, verbose=True):
        """Generate a random data.

        First, the number of alternatives of each individual is generated from
        a Poisson law or is set deterministically.
        For each alternative, utility and energy consumption are jointly
        normally distributed with zero means.
        The variance of utility and energy and the correlation coefficient are
        parameters.

        :individuals: number of individuals in the generated data, should be an
        integer greater than 2, default is 1000
        :mean_nb_alternatives: the parameter of the Poisson law used to
        generate the number of alternatives is (mean_nb_alternatives - 1),
        should be strictly greater than 1, should be an integer if use_poisson
        is false, default is 10
        :utility_variance: variance of the normal distribution for utility,
        should be positive
        :energy_variance: variance of the normal distribution for energy
        consumption, should be positive
        :correlation: correlation coefficient between utility and energy
        consumption
        :use_poisson: if True the number of alternatives is drawn from a
        modified Poisson distribution with parameter
        (mean_nb_alternatives - 1), else the number of alternatives is equal to
        mean_nb_alternatives for all individuals, default is True
        :seed: seed for the random number generator
        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert individuals > 0 and isinstance(individuals, (int, np.int64)), \
            'The parameter individuals should be an integer greater than 1'
        assert mean_nb_alternatives > 1, \
            """The parameter mean_nb_alternatives should be stricty greater
            than 1"""
        assert isinstance(mean_nb_alternatives, int) or use_poisson, \
            """The parameter mean_nb_alternatives should be an integer when
            use_poisson is false"""
        assert utility_variance >= 0, \
            'The parameter utility_variance should be positive'
        assert energy_variance >= 0, \
            'The parameter energy_variance should be positive'
        if verbose:
            # Print a progress bar of duration individuals.
            bar = _known_custom_bar(individuals, 'Generating data')
        # The Data object is generated from a random sample
        self.generate_data = True
        # Set the seed.
        np.random.seed(seed)
        if use_poisson:
            # Generate a list with the number of alternatives of all
            # individuals from a Poisson law.
            nb_alternatives = np.random.poisson(mean_nb_alternatives-1,
                                                individuals) + 1
        else:
            # All individuals have the same number of alternatives.
            nb_alternatives = np.repeat(mean_nb_alternatives, individuals)
        for i in range(individuals):
            # Generate utility and energy consumption from a bivariate normal
            # distribution.
            # The number of values generated is equal to the number of
            # alternatives of the individual i
            cov = [
                    [utility_variance, correlation],
                    [correlation, energy_variance]
            ]
            values = np.random.multivariate_normal(
                mean=[0, 0],
                cov=cov,
                size=nb_alternatives[i]
            )
            # Append the numpy array (utility, energy) to the data object.
            self._append(values)
            # Add dummy labels.
            self.alternative_labels.append(np.arange(nb_alternatives[i]))
            if verbose:
                # Update the progress bar.
                bar.update(i)
        if verbose:
            bar.finish()
        # Inform that the data were randomly generated.
        self.generated_data = True
        # Store the time spent to generate data.
        self.generating_time = time.time() - init_time

    def add_epsilons(self, scale):
        """Add a random value to the individual utilities."""
        for i in range(self.individuals):
            epsilons = np.random.normal(
                scale=scale, size=self.alternatives_per_individual[i]
            )
            self.list[i][:, 0] += epsilons

    def output_data(self, filename, delimiter=',', comment='#', verbose=True):
        """Write the data on a file.

        The output file can be read with the function read.

        :filename: string with the name of the file where the data are written
        :delimiter: the character used to separated the utility and the energy
        consumption of the alternatives, default is comma
        :comment: string used for the comments in the output file, should be a
        string, default is #
        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        # Try to open file filename and return FileNotFoundError if python is
        # unable to open the file.
        try:
            output_file = open(filename, 'wb')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if verbose:
            # Print a progress bar of duration the number of individuals.
            bar = _known_custom_bar(self.individuals, 'Writing data to file')
        # Write the data for each individual separately with the numpy.savetxt
        # command.
        for i in range(self.individuals):
            line = np.transpose(self.list[i])
            np.savetxt(output_file, line, fmt='%-7.4f',
                       header='Individual '+str(i+1), delimiter=',',
                       comments=comment)
            if verbose:
                # Update the progress bar.
                bar.update(i)
        if verbose:
            bar.finish()
        # Store the time spent to output data.
        self.output_data_time = time.time() - init_time

    def output_characteristics(self, filename, verbose=True):
        """Write a file with some characteristics on the data.

        :filename: string with the name of the file where the characteristics
        are written
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        # Try to open file filename and return FileNotFoundError if python is
        # unable to open the file.
        try:
            output_file = open(filename, 'w')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if verbose:
            print('Writing some characteristics about the data on a file...')
        size = 40
        # Display the number of individuals, the total number of alternatives,
        # the average number of alternatives, the minimum number of
        # alternatives and the maximum number of alternatives.
        output_file.write('Individuals and Alternatives'.center(size-1, '='))
        output_file.write(
                          '\nNumber of individuals:'.ljust(size)
                          + str(self.individuals)
                         )
        output_file.write(
                          '\nTotal number of alternatives:'.ljust(size)
                          + str(self.total_alternatives)
                         )
        output_file.write(
                          '\nAverage number of alternatives:'.ljust(size)
                          + str(self.total_alternatives / self.individuals)
                         )
        min_alternatives = min(self.alternatives_per_individual)
        output_file.write(
                          '\nMinimum number of alternatives:'.ljust(size)
                          + str(min_alternatives)
                         )
        max_alternatives = max(self.alternatives_per_individual)
        output_file.write(
                          '\nMaximum number of alternatives:'.ljust(size)
                          + str(max_alternatives)
                         )
        output_file.write('\n\n')
        output_file.write('Energy Consumption'.center(size-1, '='))
        # State when all the individuals choose their first alternative.
        init_state = np.zeros(self.individuals, dtype=int)
        # Show total energy consumption at the initial state.
        init_energy = self._total_energy(init_state)
        output_file.write(
                          '\nInitial energy consumption:'.ljust(size)
                          + str(init_energy)
                         )
        # State when all the individuals choose their last alternative.
        last_state = np.array(self.alternatives_per_individual) - 1
        # Show total energy consumption at the last state.
        last_energy = self._total_energy(last_state)
        output_file.write(
                          '\nMinimum potential energy consumption:'.ljust(size)
                          + str(last_energy)
                         )
        # Show the maximum energy consumption gains that could be achieved.
        max_total_energy_gains = init_energy - last_energy
        output_file.write(
                          '\nPotential energy consumption gains:'.ljust(size)
                          + str(max_total_energy_gains)
                         )
        output_file.write('\n\n')
        output_file.write('Utility and Surplus'.center(size-1, '='))
        # Show total utility at the initial state.
        init_utility = self._total_utility(init_state)
        output_file.write(
                          '\nInitial consumer surplus:'.ljust(size)
                          + str(init_utility)
                         )
        # Show total utility at the last state.
        last_utility = self._total_utility(last_state)
        output_file.write(
                          '\nMinimum consumer surplus (gross):'.ljust(size)
                          + str(last_utility)
                         )
        # Show the maximum budget that is necessary.
        max_expenses = init_utility - last_utility
        output_file.write(
                          '\nMaximum budget necessary:'.ljust(size)
                          + str(max_expenses)
                         )
        output_file.write('\n\n')
        # Indicate if the data were randomly generated or imported.
        output_file.write('Technical Information'.center(size-1, '='))
        if self.generated_data:
            output_file.write('\nThe data were randomly generated')
        else:
            output_file.write('\nThe data were imported from a file')
        # Indicate if the data are sorted.
        if self.is_sorted:
            output_file.write('\nThe data are sorted')
        else:
            output_file.write('\nThe data are not sorted')
        # Indicate if the Pareto-dominated alternatives are removed.
        if self.nb_pareto_removed:
            output_file.write('\n'
                              + str(self.nb_pareto_removed)
                              + ' Pareto-dominated alternatives were removed')
        else:
            output_file.write('\nThe Pareto-dominated alternatives are not'
                              + ' removed')
        # Store the time spent to output characteristics.
        self.output_characteristics_time = time.time() - init_time

    def remove_pareto_dominated(self, verbose=True):
        """Remove the Pareto-dominated alternatives.

        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True

        """
        # Ensure the data are sorted before cleaning.
        if not self.is_sorted:
            self.sort(verbose=verbose)
        # Store the starting time.
        init_time = time.time()
        if verbose:
            bar = _known_custom_bar(self.individuals,
                                    'Cleaning data (Pareto)')
        # Variable used to count the number of removed alternatives.
        nb_removed = 0
        # For each individual remove the Pareto-dominated alternatives.
        for indiv, line in enumerate(self.list):
            # The first alternative is never Pareto-dominated.
            sorted_line = np.array([line[0]])
            # Add the other alternatives if they are not Pareto-dominated.
            for j in range(1, len(line)):
                # The energy consumption of the alternative should be strictly
                # lower than the energy consumption of the previous non
                # Pareto-dominated alternative.
                if line[j, 1] < sorted_line[-1, 1]:
                    sorted_line = np.append(sorted_line, [line[j]], axis=0)
            # Update the array for the individual.
            self.list[indiv] = sorted_line
            # Count the number of alternatives for the individual.
            n = sorted_line.shape[0]
            # Count the number of removed alternatives.
            nb_removed += self.alternatives_per_individual[indiv] - n
            # Update the number of alternatives of the individual.
            self.alternatives_per_individual[indiv] = n
            if verbose:
                bar.update(indiv)
        # Update the total number of alternatives.
        self.total_alternatives -= nb_removed
        # Store the number of removed alternatives.
        self.nb_pareto_removed = nb_removed
        if verbose:
            bar.finish()
            print('Successfully removed '
                  + str(self.nb_pareto_removed)
                  + ' Pareto-dominated alternatives.'
                  )
        # The Pareto-dominated alternatives are now removed.
        self.pareto_dominated_removed = True
        # Store the time spent to remove the Pareto-dominated alternatives.
        self.pareto_removing_time = time.time() - init_time

    def remove_efficiency_dominated(self, python=False, verbose=True):
        """Remove the efficiency-dominated alternatives.

        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        if verbose:
            bar = _known_custom_bar(self.individuals,
                                    'Cleaning data (effic.)')
        # Variable used to count the number of removed alternatives.
        nb_removed = 0
        for i in range(self.individuals):
            if verbose:
                bar.update(i)
            # Store the alternatives of the individual.
            alternatives_list = self.list[i]
            # Compute the alternative with the lowest energy consumption among
            # the alternatives maximising utility.
            first_choice = np.lexsort(
                (alternatives_list[:, 1], -alternatives_list[:, 0])
            )[0]
            # Compute the alternative with the highest utility among the
            # alternatives minimising energy consumption.
            last_choice = np.lexsort(
                (-alternatives_list[:, 0], alternatives_list[:, 1])
            )[0]
            # If the choice with the individual best choice is also the social
            # best choice, the individual has only one relevant alternative.
            if first_choice == last_choice:
                self.list[i] = np.array([alternatives_list[first_choice]])
                self.alternative_labels[i] = np.array(
                    [self.alternative_labels[i][first_choice]])
                nb_removed += self.alternatives_per_individual[i] - 1
                self.alternatives_per_individual[i] = 1
            else:
                # If there are only two alternatives, both are non-efficiency
                # dominated.
                if self.alternatives_per_individual[i] == 2:
                    pass
                else:
                    if python:
                        self.list[i] = convex_hull(alternatives_list)
                    else:
                        try:
                            # Compute the convex hull.
                            hull = ConvexHull(alternatives_list)
                            # The vertices are the indices of the points on the
                            # convex hull.
                            v = hull.vertices
                        except Exception:
                            v = list(range(len(alternatives_list)))
                        # Compute the position in v of the first and last
                        # choice (they are always on the convex hull).
                        pos_first = [i for i, x in enumerate(v)
                                     if x == first_choice][0]
                        pos_last = [i for i, x in enumerate(v)
                                    if x == last_choice][0]
                        # The points are in counterclockwise order.
                        if pos_first < pos_last:
                            # The non-efficiency dominated choices are all
                            # points of the convex hull before pos_first and
                            # after pos_last.
                            x = v[:pos_first+1]
                            x = np.append(x, v[pos_last:])
                        else:
                            # The non-efficiency dominated choices are all
                            # points of the convex hull between pos_last and
                            # pos_first.
                            x = v[pos_last:pos_first+1]
                        if len(x) == 0:
                            print(self.list[i])
                            print(pos_first)
                            print(pos_last)
                            print(v)
                            print('\n')
                        # Update the array for the individual.
                        self.list[i] = alternatives_list[x]
                        self.alternative_labels[i] = \
                            self.alternative_labels[i][x]
                    # Count the number of alternatives for the individual.
                    n = self.list[i].shape[0]
                    # Count the number of removed alternatives.
                    nb_removed += self.alternatives_per_individual[i] - n
                    # Update the number of alternatives of the individual.
                    self.alternatives_per_individual[i] = n
        # Update the total number of alternatives.
        self.total_alternatives -= nb_removed
        # Store the number of removed alternatives.
        self.nb_efficiency_removed = nb_removed
        if verbose:
            bar.finish()
            print('Successfully removed '
                  + str(self.nb_efficiency_removed)
                  + ' efficiency-dominated alternatives.'
                  )
        # The efficiency-dominated alternatives are now removed.
        self.efficiency_dominated_removed = True
        if not python:
            # The Pareto-dominate alternatives are also removed, by definition.
            self.pareto_dominated_removed = True
        # The data are now unsorted.
        self.is_sorted = False
        # Store the time spent to remove the Pareto-dominated alternatives.
        self.efficiency_removing_time = time.time() - init_time

    def sort(self, verbose=True):
        """Sort the data by utility, then by energy consumption.

        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True

        """
        # Store the starting time.
        init_time = time.time()
        if verbose:
            bar = _known_custom_bar(self.individuals,
                                    'Sorting data')
        for indiv, line in enumerate(self.list):
            # The numpy array for each individual is sorted by utility and then
            # by energy consumption.
            sorting_indices = np.lexsort((line[:, 1], -line[:, 0]))
            sorted_line = line[sorting_indices]
            self.list[indiv] = sorted_line
            if len(self.alternative_labels) > 0:
                # Also sort labels.
                sorted_labels = self.alternative_labels[indiv][sorting_indices]
                self.alternative_labels[indiv] = sorted_labels
            if verbose:
                bar.update(indiv)
        if verbose:
            bar.finish()
        # The data are now sorted.
        self.is_sorted = True
        # Store the time spent to sort data.
        self.sorting_time = time.time() - init_time

    def _total_utility(self, state):
        """Compute the sum of utilities of a given state.

        :state: a list with all the alternatives of the individuals
        :returns: float indicating the sum of utilities

        """
        total = np.sum(np.array(
                    [array[state[i], 0] for i, array in enumerate(self.list)]
                ))
        return total

    def _total_energy(self, state):
        """Compute the total energy consumption of a given state.

        :state: a list with all the alternatives of the individuals
        :returns: float indicating the total energy consumption

        """
        total = sum([array[state[i], 1] for i, array in enumerate(self.list)])
        return total

    def _get_nb_alternatives(self, individual):
        """Return the number of alternatives of the specified individual.

        :individual: index of the individual considered, should be an integer
        :returns: integer indicating the number of alternatives

        """
        J = self.alternatives_per_individual[individual]
        return J

    def _get_utility(self, individual):
        """Return the array of the utility of the alternatives of the specified
        individual.

        :individual: index of the individual considered, should be an integer
        :returns: array with the utility of the alternatives

        """
        array = self.list[individual][:, 0]
        return array

    def _get_energy(self, individual):
        """Return the array of the energy consumption of the alternatives of
        the specified individual.

        :individual: index of the individual considered, should be an integer
        :returns: array with the energy consumption of the alternatives

        """
        array = self.list[individual][:, 1]
        return array

    def _single_jump_efficiency(self, individual, previous_alternative,
                                next_alternative):
        """Compute the efficiency of the jump from one alternative to another
        alternative for a specific individual.

        :individual: the index of the considered individual, should be an
        integer
        :previous_alternative: the index of the previous alternative of the
         individual, should be an integer
        :next_alternative: the index of the previous alternative of the
         individual, should be an integer
        :returns: a float indicating the efficiency of the jump

        """
        # If the previous alternative is greater than the next alternative (in
        # terms of individual utility) then the efficiency is 0.
        if previous_alternative >= next_alternative:
            efficiency = 0
        else:
            # Take the utility and the energy consumption of the alternatives
            # of the individual.
            utility = self._get_utility(individual)
            energy = self._get_energy(individual)
            # Save the value of utility and energy consumption for the
            # alternative of the individual.
            previous_utility = utility[previous_alternative]
            previous_energy = energy[previous_alternative]
            # Compute the efficiency.
            efficiency = (previous_energy - energy[next_alternative]) \
                / (previous_utility - utility[next_alternative])
        return efficiency

    def _individual_jump_efficiencies(self, individual, choice, results):
        """Compute the efficiency of all the possible jumps of a single
        individual.

        The efficiency of a jump is defined as the difference in energy
        consumption between two alternatives divided by the difference in
        utility between these two alternatives.
        The possible jumps are all the jumps going from the choice of the
        individual to any other alternatives with a lower individual utility.
        An array is returned with the same length as the number of alternatives
        of the considered individual.
        By convention, the efficiency is 0 if the jump is not possible.

        :individual: the index of the considered individual, should be an
         integer
        :choice: the index of the choice of the individual, should be an
        integer
        :results: an AlgorithmResults object
        :returns: a numpy array with the efficiencies for all the possible
        jumps

        """
        # Take the number of alternatives of the individual.
        J = self._get_nb_alternatives(individual)
        if choice == J-1:
            # The individual is already at best alternative.
            efficiencies = np.zeros(1)
        else:
            # Take the utility and the energy consumption of the alternatives
            # of the individual.
            utility = self._get_utility(individual)
            energy = self._get_energy(individual)
            # Save the value of utility and energy consumption for the choice
            # of the individual.
            current_utility = utility[choice]
            current_energy = energy[choice]
            # Compute the incentive amount needed and the energy gains of the
            # jumps to a next alternative.
            energy_gains = current_energy - energy[choice+1:]
            incentive_amount = current_utility - utility[choice+1:]
            if results.small_jumps:
                remaining_budget = results.budget - results.expenses
                # Count the number of jumps within the budget.
                within_budget = incentive_amount <= remaining_budget
                nb_possible_jumps = sum(within_budget)
                if nb_possible_jumps == 0:
                    # There are no jump possible.
                    efficiencies = np.zeros(1)
                else:
                    # Compute the efficiencies.
                    energy_gains = energy_gains[within_budget]
                    incentive_amount = incentive_amount[within_budget]
                    efficiencies = energy_gains / incentive_amount
            else:
                # Compute the efficiencies.
                efficiencies = energy_gains / incentive_amount
        return efficiencies

    def _individual_best_jump(self, individual, choice, results):
        """Compute the best jump of the individual.

        :individual: the index of the considered individual, should be an
        integer
        :choice: the index of the choice of the individual, should be an
        integer
        :results: an AlgorithmResults object
        :returns: an integer indicating the resulting alternative of the best
         jump and a float indicating the efficiency of the best jump

        """
        # Compute the efficiency of all the jumps and take the maximum.
        efficiencies = self._individual_jump_efficiencies(
                individual, choice, results
        )
        max_efficiency = np.max(efficiencies)
        # If the value of the maximum efficiency is 0, that means that no jump
        # is possible so the individual is already at its best alternative.
        if max_efficiency == 0:
            best_alternative = 0
        # Else, the best alternative is the resulting alternative of the jump
        # with the highest efficiency.
        # In case of ties, python automatically chooses the jump with the lower
        # incentive.
        else:
            best_alternative = choice + 1 + np.argmax(efficiencies)
        return max_efficiency, best_alternative

    def _all_best_jump(self, results):
        """Compute the best jump of all individuals and return an array with
        the resulting alternatives and the efficiencies.

        :results: an AlgorithmResults object
        :returns: a numpy array with one column per individual and two rows
        (efficiency and resulting alternative of the jumps)

        """
        state = results.optimal_state
        efficiencies = []
        alternatives = []
        # For each individual, compute the resulting alternative and the
        # efficiency of the best jump and append the results to the lists.
        for i in range(self.individuals):
            efficiency, alternative = self._individual_best_jump(
                    i, state[i], results
            )
            efficiencies.append(efficiency)
            alternatives.append(alternative)
        efficiencies = np.array(efficiencies)
        return efficiencies, alternatives

    def _incentives_amount(self, individual, previous_alternative,
                           next_alternative):
        """Compute the amount of incentives needed to induce the individual to
        change his choice.

        :individual: the index of the considered individual, should be an
         integer
        :previous_alternative: the index of the alternative currently chosen by
         the individual, should be an integer
        :next_alternative: the index of the targeted alternative, should be an
         integer
        :returns: a float indicating the amount of incentives of the jump

        """
        # Get the utility of the individual at his previous alternative and his
        # next alternative.
        utilities = self._get_utility(individual)
        previous_utility = utilities[previous_alternative]
        next_utility = utilities[next_alternative]
        # The amount of incentives is defined by the loss in utility.
        incentives = previous_utility - next_utility
        return incentives

    def _energy_gains_amount(self, individual, previous_alternative,
                             next_alternative):
        """Compute the amount of energy gains when the specified individual
        change his choice.

        :individual: the index of the considered individual, should be an
        integer
        :previous_alternative: the index of the alternative currently chosen by
         the individual, should be an integer
        :next_alternative: the index of the targeted alternative, should be an
         integer
        :returns: a float indicating the amount of energy gains of the jump

        """
        # Get the enery consumption of the individual at his previous
        # alternative and his next alternative.
        energy = self._get_energy(individual)
        previous_energy = energy[previous_alternative]
        next_energy = energy[next_alternative]
        # Compute the amount of energy gains.
        energy_gains = previous_energy - next_energy
        return energy_gains

    def get_choices(self, incentives, init_state, last_state):
        """Return a list of booleans. True if the individual accept the
        incentive.
        """
        choices = list()
        for i in range(self.individuals):
            init_utility = self.list[i][init_state[i], 0]
            last_utility = self.list[i][last_state[i], 0]
            if (init_utility - last_utility) < incentives[i]:
                choices.append(True)
            else:
                choices.append(False)
        return np.array(choices)

    def run_algorithm(self, budget=np.infty, force=True, remove_pareto=True,
                      small_jumps=False, verbose=True):
        """Compute the optimal state for a given budget by running the
        algorithm.

        If the available budget is enough to reach the state where all
        individuals are at their last alternative and if force is False, then
        the algorithm is not run and the state is directly returned.

        :budget: should be an integer or a float with the maximum amount of
        incentives to give, default is np.infty (the budget is unlimited).
        :force: if True, force the algorithm to run even if it is not
        necessary, default is True.
        :remove_pareto: if True, remove the Pareto-dominated alternatives
        before running the algorithm
        :small_jumps: if True, use a different version of the algorithm with
        jumps constrained to be within the budget
        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True
        :returns: an AlgorithmResults object

        """
        # The data must be sorted.
        if not self.is_sorted:
            self.sort(verbose=verbose)
        # Remove the Pareto-dominated alternatives.
        if not self.pareto_dominated_removed and remove_pareto:
            self.remove_pareto_dominated(verbose=verbose)
        # Store the starting time.
        init_time = time.time()
        if verbose:
            if budget == np.infty:
                bar = _known_custom_bar(
                    self.total_alternatives - self.individuals,
                    'Running algorithm'
                )
            else:
                bar = _known_custom_bar(budget,
                                        'Running algorithm')
        # Create an AlgorithmResults object where the variables are stored.
        results = AlgorithmResults(self, budget)
        results.small_jumps = small_jumps
        # Compute the amount of expenses needed to reach the state where all
        # individuals are at their last alternative.
        # Return the last state if the budget is enough to reach it.
        if (not force) and (budget >= results.max_expenses):
            results.optimal_state = results.last_state
            results.expenses = results.max_expenses
            if verbose:
                bar.finish()
        else:
            # Compute the efficiency and the resulting alternative of the best
            # jump of all individuals.
            best_efficiencies, best_alternatives = self._all_best_jump(results)
            # Main loop of the algorithm.
            # The loop runs until the budget is depleted or until all the jumps
            # have been done.
            while results.expenses < budget and \
                    sum(best_efficiencies != 0) != 0:
                if verbose:
                    if budget == np.infty:
                        bar.update(np.sum(results.optimal_state))
                    else:
                        bar.update(results.expenses)
                # Increase the number of iterations by 1.
                results.iteration += 1
                # Select the individual with the most efficient jump.
                selected_individual = np.argmax(best_efficiencies)
                # Store information on the jump (selected individual, previous
                # alternative, next alternative).
                previous_alternative = \
                    results.optimal_state[selected_individual]
                next_alternative = best_alternatives[selected_individual]
                jump_information = [selected_individual,
                                    previous_alternative,
                                    next_alternative]
                results.jumps_history.append(jump_information)
                # Store the efficiency of the jump.
                jump_efficiency = best_efficiencies[selected_individual]
                results.efficiencies_history.append(jump_efficiency)
                # Change the current state according to the new choice of the
                # selected individual.
                results.optimal_state[selected_individual] = next_alternative
                # Update the arrays of the best jumps for the selected
                # individual.
                new_best_efficiency, new_best_alternative = \
                    self._individual_best_jump(
                            selected_individual, next_alternative, results
                    )
                best_efficiencies[selected_individual] = new_best_efficiency
                best_alternatives[selected_individual] = new_best_alternative
                # Increase the expenses by the amount of incentives of the
                # jump.
                incentives = self._incentives_amount(*jump_information)
                results.expenses += incentives
                # With the "small jumps" technique, update the best jumps if
                # they exceed the budget.
                if small_jumps:
                    remaining_budget = results.budget - results.expenses
                    for i in range(self.individuals):
                        jump_information = [
                                i,
                                results.optimal_state[i],
                                best_alternatives[i]
                        ]
                        cost = self._incentives_amount(*jump_information)
                        if cost > remaining_budget:
                            new_best_efficiency, new_best_alternative = \
                                self._individual_best_jump(
                                        i, results.optimal_state[i], results
                                )
                            best_efficiencies[i] = new_best_efficiency
                            best_alternatives[i] = new_best_alternative
                # Increase the total energy gains by the amount of energy gains
                # of the jump.
                energy_gains = incentives * jump_efficiency
                results.total_energy_gains += energy_gains
                # Store the incentives and the energy gains of the jump.
                results.incentives_history.append(incentives)
                results.energy_gains_history.append(energy_gains)
            # In case of overshot (the expenses are greater than the budget),
            # go back to the previous iteration.
            if results.expenses > budget:
                # Reduce the number of iterations by 1.
                results.iteration -= 1
                # Restore the previous state.
                results.optimal_state[selected_individual] = \
                    previous_alternative
                # Reduce the expenses by the amount of incentives of the last
                # jump.
                results.expenses -= incentives
                # Reduce the total energy gains by the amount of energy gains
                # of the last jump.
                results.total_energy_gains -= energy_gains
                # Remove the last jump of the list and store it in a special
                # variable.
                results.overshot_jump = results.jumps_history[-1]
                results.jumps_history = results.jumps_history[:-1]
                # Remove the last value of incentives, energy gains and
                # efficiency.
                results.efficiencies_history = \
                    results.efficiencies_history[:-1]
                results.incentives_history = results.incentives_history[:-1]
                results.energy_gains_history = \
                    results.energy_gains_history[:-1]
            # Indicate that the algorithm was fully run.
            results.run_algorithm = True
        if verbose:
            bar.finish()
        # Store the time spent to run the algorithm.
        results.algorithm_running_time = time.time() - init_time
        return results

    def run_lite_algorithm(self, budget=np.infty, force=True, verbose=True):
        """Compute the optimal state for a given budget by running a modified
        version of the Algorithm, usable only when the efficiency-dominated
        alternatives are removed.

        If the available budget is enough to reach the state where all
        individuals are at their last alternative and if force is False, then
        the algorithm is not run and the state is directly returned.

        :budget: should be an integer or a float with the maximum amount of
        incentives to give, default is np.infty (the budget is unlimited).
        :force: if True, force the algorithm to run even if it is not
        necessary, default is True.
        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True
        :returns: an AlgorithmResults object

        """
        # Running the lite algorithm only work if the efficiency-dominated
        # alternatives are removed.
        if not self.efficiency_dominated_removed:
            self.remove_efficiency_dominated(verbose=verbose)
        if not self.pareto_dominated_removed:
            self.remove_pareto_dominated(verbose=verbose)
        # The data must be sorted.
        if not self.is_sorted:
            self.sort(verbose=verbose)
        # Store the starting time.
        init_time = time.time()
        if verbose:
            if budget == np.infty:
                bar = _known_custom_bar(
                    self.total_alternatives - self.individuals,
                    'Running algorithm'
                )
            else:
                bar = _known_custom_bar(budget,
                                        'Running algorithm')
        # Create an AlgorithmResults object where the variables are stored.
        results = AlgorithmResults(self, budget)
        # Compute the amount of expenses needed to reach the state where all
        # individuals are at their last alternative.
        # Return the last state if the budget is enough to reach it.
        if (not force) and (budget >= results.max_expenses):
            results.optimal_state = results.last_state
            results.expenses = results.max_expenses
            if verbose:
                bar.finish()
        else:
            # Compute the array jump_list with informations on all the possible
            # jumps (individual, initial alternative, incentive amount, energy
            # gains and efficiency).
            # The total number of jumps is the number of alternatives minus 1
            # for each individual.
            total_nb_jump = self.total_alternatives-self.individuals
            # Create an empty array and a counter used to fill the empty array.
            jump_list = np.empty([total_nb_jump, 5])
            J = 0
            for i in range(self.individuals):
                # Compute 5 lists with informations on the jumps of the
                # individual.
                nb_jump = self.alternatives_per_individual[i]-1
                individual = np.repeat(i, nb_jump)
                initial_alternatives = np.arange(0, nb_jump, 1)
                incentives = np.array(
                        [self._incentives_amount(i, j, j+1)
                            for j in initial_alternatives]
                        )
                energy_gains = np.array(
                        [self._energy_gains_amount(i, j, j+1)
                            for j in initial_alternatives]
                        )
                efficiencies = energy_gains / incentives
                # Fill the jump_list array with the computed lists.
                jump_list[J:J+nb_jump] = np.transpose(np.array(
                    [individual,
                     initial_alternatives,
                     incentives,
                     energy_gains,
                     efficiencies]
                ))
                # Increase the counter.
                J += nb_jump
            # Sort the jump_list array by efficiency.
            jump_list = jump_list[jump_list[:, 4].argsort()[::-1]]
            # The expenses are the cumulative sum of the incentives.
            expenses = np.cumsum(jump_list[:, 2])
            # Search after how many jumps the budget is depleted.
            results.iteration = np.searchsorted(expenses, budget)
            # Store informations on the jumps.
            selected_individuals = jump_list[:results.iteration, 0].astype(int)
            previous_alternatives = \
                jump_list[:results.iteration, 1].astype(int)
            next_alternatives = previous_alternatives + 1
            results.jumps_history = np.transpose(np.array(
                    [selected_individuals,
                     previous_alternatives,
                     next_alternatives]
            ))
            results.incentives_history = jump_list[:results.iteration, 2]
            results.energy_gains_history = jump_list[:results.iteration, 3]
            results.efficiencies_history = jump_list[:results.iteration, 4]
            # To compute the optimal state, count the number of occurrence of
            # each individual in the selected_individuals list.
            count = np.unique(selected_individuals, return_counts=True)
            # Change the optimal state to the number of occurrence for
            # individuals who moved. The value stays at 0 for individual who
            # did not moved.
            results.optimal_state[count[0]] = count[1]
            # Compute the expenses and total energy gains using the cumulative
            # sum of the incentives and the energy gains.
            results.expenses = expenses[results.iteration-1]
            energy_gains = np.cumsum(jump_list[:results.iteration, 3])
            results.total_energy_gains = energy_gains[-1]
            if budget < results.max_expenses:
                # Store the next jump.
                selected_individuals = \
                    jump_list[results.iteration, 0].astype(int)
                previous_alternatives = \
                    jump_list[results.iteration, 1].astype(int)
                next_alternatives = previous_alternatives + 1
                results.overshot_jump = np.transpose(np.array(
                        [selected_individuals,
                         previous_alternatives,
                         next_alternatives]
                ))
            # Indicate that the algorithm was fully run.
            results.run_algorithm = True
        if verbose:
            bar.finish()
        # Store the time spent to run the algorithm.
        results.algorithm_running_time = time.time() - init_time
        return results

    def find_optimum(self, budget=np.infty, verbose=True):
        """Search for the optimum state by computing the energy of all the
        states.

        This function can take a very long time to run with many individuals
        and alternatives.

        :budget: should be an integer or a float with the maximum amount of
        incentives to give, default is np.infty (the budget is unlimited).
        :verbose: if True, a progress bar and some information are displayed
        during the process, default is True.
        :returns: the optimal state, the optimal energy gains and the cost.
        """
        # Remove the Pareto-dominated alternatives.
        if not self.pareto_dominated_removed:
            self.remove_pareto_dominated(verbose=verbose)
        # Build a list of all the possible states.
        alternative_list = [
                list(range(j)) for j in self.alternatives_per_individual
        ]
        state_list = itertools.product(*alternative_list)
        # Compute initial utility.
        init_state = np.zeros(self.individuals, dtype=int)
        init_utility = self._total_utility(init_state)
        if verbose:
            bar = _known_custom_bar(
                np.prod(self.alternatives_per_individual),
                'Finding optimum'
            )
        # If the state is not too costly, add its energy to the list.
        reachable_states = []
        for i, state in enumerate(state_list):
            if verbose:
                bar.update(i)
            cost = init_utility - self._total_utility(state)
            if cost <= budget:
                energy = -self._total_energy(state)
                reachable_states.append((list(state), energy, cost))
        # Build a structured numpy array.
        dtype = [
                ('state', int, self.individuals),
                ('energy', float),
                ('cost', float)
        ]
        reachable_states = np.array(reachable_states, dtype=dtype)
        if budget == np.infty:
            # Sort the states by energy.
            reachable_states = np.sort(
                    reachable_states,
                    order=['cost', 'energy']
            )
            # Remove the Pareto-dominated states.
            optimum = np.array([reachable_states[0]])
            for j in range(1, len(reachable_states)):
                if reachable_states[j]['energy'] > optimum[-1]['energy']:
                    optimum = np.append(
                            optimum,
                            [reachable_states[j]],
                            axis=0
                    )
        else:
            # Find the state with the highest energy gains among the reachable
            # states.
            optimal_state_index = np.argmax(reachable_states['energy'])
            optimum = reachable_states[optimal_state_index]
        if verbose:
            bar.finish()
        return optimum

    def run_optimal_algorithm(self, budget=np.infty, decimals=0, verbose=True):
        """Run the algorithm to find the optimum."""
        # The data must be sorted.
        if not self.is_sorted:
            self.sort(verbose=verbose)
        # Remove the Pareto-dominated alternatives.
        if not self.pareto_dominated_removed:
            self.remove_pareto_dominated(verbose=verbose)
        # Compute arrays with the cost and increase in social utility of all
        # the alternatives.
        self.diff_list = []
        for array in self.list:
            # Compute the difference between the initial values and the values
            # of the alternatives.
            diff_array = array[0] - array
            # Round the jump cost to the decimal value.
            diff_array[:, 0] = np.round(diff_array[:, 0], 2)
            self.diff_list.append(diff_array)
        # The budget must not be greater than the maximum possible expenses.
        results = AlgorithmResults(self, budget)
        if results.max_expenses < budget:
            budget = results.max_expenses
        # The algorithm find the optimum for all values of the budget in
        # budget_list.
        interval = 10**(-decimals)
        budget_list = np.arange(0, budget+interval, interval)
        # Store the optimal values in a numpy array.
        # There are as many optimum values as there are budgets.
        B = len(budget_list)
        optimums = np.zeros(B)
        if verbose:
            bar = _known_custom_bar(
                self.individuals,
                'Running algorithm'
            )
        for i in range(self.individuals):
            if verbose:
                bar.update(i)
            old_optimums = np.copy(optimums)
            for j in range(1, self.alternatives_per_individual[i]):
                cost, utility = self.diff_list[i][j]
                cost_position = int(cost / interval)
                gains = utility + old_optimums[:B-cost_position]
                optimums[cost_position:] = np.maximum(
                        gains, optimums[cost_position:]
                )
        if verbose:
            bar.finish()
        return optimums


class KnapsackData:

    def __init__(self, data):
        self.individuals = data.individuals
        self.list = data.list.copy()
        self.list = convert_to_knapsack(self.list)
        self.alternatives_per_individual = \
            data.alternatives_per_individual.copy()

    def run_dyer_zemel(self, budget=0, verbose=True):
        """Implementation of Dyer-Zemel algorithm.
        """

        if verbose:
            print('Starting Dyer-Zemel algorithm')
        not_finished = True

        # 0.
        # Keep track of individuals with only one alternative.
        non_empty_individuals = list(np.arange(self.individuals, dtype=int))
        for i in non_empty_individuals:
            if len(self.list[i]) == 1:
                # Set individual as empty.
                non_empty_individuals.remove(i)
                # Decrease budget by weight of remaining alternative.
                choice = 0
                budget -= self.list[i][choice][0]
        # Create empty sets of removed alternatives.
        removed = [set() for i in range(self.individuals)]

        while not_finished:

            # 1.
            # Create pairs of alternatives.
            new_indexes = [[] for i in range(self.individuals)]
            for i in non_empty_individuals:
                # Store the value and weight of the individual's alternatives.
                alternatives = self.list[i]
                # Store the indexes of the alternatives of the individual.
                indexes = set(range(self.alternatives_per_individual[i]))
                indexes = indexes.difference(removed[i])
                indexes = list(indexes)
                unsorted = True
                while unsorted:
                    if len(indexes) > 1:
                        # Build a pair with the two alternatives whose index
                        # are first in the index set.
                        pair_indexes = (indexes[0], indexes[1])
                        pair = (alternatives[indexes[0]],
                                alternatives[indexes[1]])
                        # Order the pair.
                        if pair[0][0] > pair[1][0]:
                            # First element has higher weight so the pair must
                            # be inverted.
                            pair_indexes = pair_indexes[::-1]
                            pair = pair[::-1]
                        elif pair[0][0] == pair[1][0]:
                            # Check values.
                            if pair[0][1] < pair[1][1]:
                                # First element has lower value so the pair
                                # must be inverted.
                                pair_indexes = pair_indexes[::-1]
                                pair = pair[::-1]
                        if pair[0][0] < pair[1][0] and pair[0][1] > pair[1][1]:
                            # The second element is dominated, remove it from
                            # the indexes.
                            removed[i].add(pair_indexes[1])
                            indexes.remove(pair_indexes[1])
                        else:
                            # Store the indexes of the pair.
                            new_indexes[i].append(pair_indexes[0])
                            new_indexes[i].append(pair_indexes[1])
                            # Remove the indexes of the alternatives.
                            indexes.remove(pair_indexes[0])
                            indexes.remove(pair_indexes[1])
                    else:
                        # No more pairs can be built.
                        unsorted = False
                        if len(indexes) == 1:
                            # Add the last alternative.
                            new_indexes[i].append(indexes[0])

            # 2.
            # Individuals with only one non-dominated alternative.
            for i in non_empty_individuals:
                if len(new_indexes[i]) == 1:
                    # Set individual as empty.
                    non_empty_individuals.remove(i)
                    # Decrease budget by weight of remaining alternative.
                    choice = new_indexes[i][0]
                    budget -= self.list[i][choice][0]

            # 3.
            # Compute the slope of all the pairs and find the median.
            pairs = []
            slopes = []
            for i in non_empty_individuals:
                alternatives = self.list[i]
                nb_pairs = int(len(new_indexes[i]) / 2)
                for j in range(nb_pairs):
                    # Retrieve the pair from the indexes.
                    first_index = new_indexes[i][2*j]
                    second_index = new_indexes[i][2*j+1]
                    first = alternatives[first_index]
                    second = alternatives[second_index]
                    # Compute the slope.
                    slope = (second[1]-first[1]) / (second[0]-first[0])
                    pairs.append([i, first_index, second_index, slope])
                    slopes.append(slope)
            slopes = np.array(slopes)
            median_slope = np.median(slopes)
            print('median slope: {}'.format(median_slope))

            # 4.
            min_weights = dict()
            max_weights = dict()
            for i in non_empty_individuals:
                alternatives = self.list[i]
                most_extremes = list()
                max_value = -np.inf
                values = alternatives[:, 1] - median_slope * alternatives[:, 0]
                for index in new_indexes[i]:
                    if values[index] > max_value:
                        max_value = values[index]
                        most_extremes = list()
                        most_extremes.append(index)
                    elif values[index] == max_value:
                        most_extremes.append(index)
                weights = alternatives[:, 0]
                most_extremes = np.array(most_extremes, dtype=int)
                # Find index of alternative with min weight in most_extremes
                # array.
                mini = np.argmin(weights[most_extremes])
                # Retrieve true index of alternative.
                min_weights[i] = most_extremes[mini]
                maxi = np.argmax(weights[most_extremes])
                max_weights[i] = most_extremes[maxi]

            # 5.
            lower_sum = 0
            upper_sum = 0
            for i in non_empty_individuals:
                min_alternative = self.list[i][min_weights[i]]
                max_alternative = self.list[i][max_weights[i]]
                lower_sum += min_alternative[0]
                upper_sum += max_alternative[0]
            if lower_sum > budget:
                for pair in pairs:
                    if pair[3] <= median_slope:
                        # Remove k.
                        removed[pair[0]].add(pair[2])
            elif upper_sum <= budget:
                for pair in pairs:
                    if pair[3] >= median_slope:
                        # Remove j.
                        removed[pair[0]].add(pair[1])
            else:
                print('Finished')
                print('Result: {}'.format(median_slope))
                not_finished = False


class AlgorithmResults:

    """An AlgorithmResults object is used to store the output generated by the
    run_algorithm method.

    """

    def __init__(self, data, budget):
        """Initiate an AlgorithmResults object. """
        # Store the budget and the Data object associated with the
        # AlgorithmResults object.
        self.data = data
        self.budget = budget
        # Boolean to indicate if "small jumps" technique is enabled.
        self.small_jumps = False
        # Compute the initial state and the first best state.
        self.init_state = np.zeros(self.data.individuals, dtype=int)
        self.last_state = np.array(self.data.alternatives_per_individual) - 1
        # Compute initial utility, first best utility and max
        # budget necessary.
        self.init_utility = self.data._total_utility(self.init_state)
        self.last_utility = self.data._total_utility(self.last_state)
        self.max_expenses = self.init_utility - self.last_utility
        self.percent_max_expenses = self.budget / self.max_expenses
        # Same for energy.
        self.init_energy = self.data._total_energy(self.init_state)
        self.last_energy = self.data._total_energy(self.last_state)
        self.max_total_energy_gains = self.init_energy - self.last_energy
        # The variable optimal_state is a numpy array with the alternative
        # chosen by all the individuals at the optimal state, the array is
        # initialized with all individuals being at their first alternative.
        self.optimal_state = np.zeros(data.individuals, dtype=int)
        # The variable iteration counts the number of iterations of the
        # algorithm.
        self.iteration = 0
        # The variable expenses is equal to the total amount of incentives
        # needed to reach the optimal state.
        self.expenses = 0
        # The variable total_energy_gains is equal to the decrease in energy
        # consumption at the optimal state compared to the initial state.
        self.total_energy_gains = 0
        # At each iteration, the selected individual, the previous alternative
        # and the next alternative are stored in the variable jumps_history.
        # There are as many rows as iterations.
        # There are three columns: selected individual, previous jump and next
        # jump.
        self.jumps_history = []
        # The variable incentives_history is a list where the amount of
        # incentives for each iteration is stored.
        self.incentives_history = []
        # The variable energy_gains_history is a list where the amount of
        # energy gains for each iteration is stored.
        self.energy_gains_history = []
        # The variable efficiencies_history is a list where the efficiency of
        # each jump is stored.
        self.efficiencies_history = []
        # The variable overshot_jump is used to store information on the
        # overshot jump (the jump that is removed in case of overshot).
        self.overshot_jump = None
        # The variable run_algorithm is True if the algorithm was fully run.
        self.run_algorithm = False
        # The variable computed_results is True if the method compute_results
        # was run.
        self.computed_results = False
        # Generate empty variables to store the computing times.
        self.algorithm_running_time = None
        self.computing_results_time = None
        self.output_results_time = None
        self.output_characteristic_time = None

    def compute_results(self, verbose=True):
        """Compute relevant results from the raw results of the algorithm.

        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert self.run_algorithm, \
            'The algorithm was not run.'
        if verbose:
            print('Computing additional results...')
        self.jumps_history = np.array(self.jumps_history)
        # Compute the incentives needed to move the individuals from their
        # first alternative to their optimal alternative.
        self.total_incentives = [
            self.data._incentives_amount(i, 0, self.optimal_state[i])
            for i in range(self.data.individuals)
        ]
        # Compute optimal utility and optimal energy gains.
        self.optimal_utility = self.init_utility - self.expenses
        self.optimal_energy = self.init_energy - self.total_energy_gains
        # Compute the remaining budget and the percentage of budget used.
        self.budget_remaining = self.budget - self.expenses
        self.percent_budget = self.expenses / self.budget
        # Compute the energy gains as percentage of total energy gains
        # possible.
        self.percent_total_energy_gains = \
            self.total_energy_gains / self.max_total_energy_gains
        # Compute efficiency of first and last jump and total efficiency.
        self.first_efficiency = self.efficiencies_history[0]
        self.last_efficiency = self.efficiencies_history[-1]
        self.optimal_state_efficiency = self.total_energy_gains / self.expenses
        # Compute the maximum, minimum and average amount of incentives of a
        # jump.
        self.max_incentives = max(self.incentives_history)
        self.min_incentives = min(self.incentives_history)
        self.average_incentives = np.mean(self.incentives_history)
        # Compute the maximum, minimum and average energy gains of a jump.
        self.max_energy_gains = max(self.energy_gains_history)
        self.min_energy_gains = min(self.energy_gains_history)
        self.average_energy_gains = np.mean(self.energy_gains_history)
        # Compute the average number of jumps (= iterations).
        self.average_nb_jumps = self.iteration / self.data.individuals
        # Compute the percentage of individuals at first and last alternative.
        self.percent_at_first_alternative = np.sum(self.optimal_state == 0) \
            / self.data.individuals
        self.percent_at_last_alternative = np.sum(
            self.optimal_state ==
            np.array(self.data.alternatives_per_individual)-1
        ) / self.data.individuals
        # Compute the bound interval and the bound.
        if self.overshot_jump is None:
            self.bound_size = 0
        else:
            self.bound_size = (
                self.budget_remaining
                * self.data._energy_gains_amount(
                    self.overshot_jump[0],
                    self.overshot_jump[1],
                    self.overshot_jump[2]
                )
                / self.data._incentives_amount(
                    self.overshot_jump[0],
                    self.overshot_jump[1],
                    self.overshot_jump[2]
                )
            )
        self.max_bound = self.optimal_energy
        self.min_bound = self.max_bound - self.bound_size
        # Compute the amount of expenses at each iteration.
        self.expenses_history = \
            np.append(0, np.cumsum(self.incentives_history))
        # Compute the total energy gains at each iteration.
        self.total_energy_gains_history = \
            np.append(0, np.cumsum(self.energy_gains_history))
        self.energy_history = (
            self.init_energy
            - self.total_energy_gains_history
        )
        # Compute the bound differences.
        self.bound_differences = \
            np.append(np.diff(self.total_energy_gains_history), 0)
        # Compute an array going from 0 to the number of iterations.
        self.iterations_history = np.arange(0, self.iteration, 1)
        # To compute the number of individuals at their first alternative for
        # each iteration, we build an array where each element is 1 if the
        # previous alternative of the associated jump is 0 and else is 0. The
        # cumulative sum of this array gives the number of individuals NOT at
        # their first alternative. The number of individuals at their first
        # alternative is the total number of individuals minus the array we
        # computed.
        x = np.zeros(self.iteration, dtype=int)
        y = np.array(self.jumps_history)[:, 1]
        x[y == 0] = 1
        self.moved_history = np.cumsum(x)
        self.at_first_alternative_history = \
            self.data.individuals - self.moved_history
        # To compute the number of individuals at their last alternative, we
        # use the same method but the elements of the first array is 1 if the
        # next alternative of the jump is equal to the last jump of the
        # associated individual. We also need to add the number of individuals
        # with one alternative (they are always at their last alternative).
        x = np.zeros(self.iteration, dtype=int)
        y = np.array(self.jumps_history)[:, 2]
        z = np.array(self.jumps_history)[:, 0]
        z = [self.last_state[i] for i in z]
        x[y == z] = 1
        x = np.cumsum(x)
        w = np.sum(self.last_state == 0)
        self.at_last_alternative_history = x + w
        # Compute the share of each choice at each iteration.
        unique_labels = {label
                         for labels in self.data.alternative_labels
                         for label in labels}
        unique_labels = np.sort(list(unique_labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        nb_choices = np.zeros(unique_labels.shape, dtype=np.int64)
        init_choices = np.array(
            [labels[0] for labels in self.data.alternative_labels])
        uniques, counts = np.unique(init_choices, return_counts=True)
        nb_choices[np.isin(unique_labels, uniques)] = counts
        choice_shares = [nb_choices.copy()]
        for jump in self.jumps_history:
            old_choice = self.data.alternative_labels[jump[0]][jump[1]]
            old_choice_id = label_to_id[old_choice]
            new_choice = self.data.alternative_labels[jump[0]][jump[2]]
            new_choice_id = label_to_id[new_choice]
            nb_choices[old_choice_id] -= 1
            nb_choices[new_choice_id] += 1
            choice_shares.append(nb_choices.copy())
        self.choice_shares = (
            np.transpose(choice_shares) / self.data.individuals
        )
        self.labels = unique_labels
        # Inform that the results were computed.
        self.computed_results = True
        # Store the time spent computing results.
        self.computing_results_time = time.time() - init_time

    def output_characteristics(self, filename, verbose=True):
        """Write a file with some characteristics on the results of the
        algorithm.

        :filename: string with the name of the file where the information are
        written
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert self.run_algorithm, \
            'The algorithm was not run.'
        # Try to open file filename and return FileNotFoundError if python is
        # unable to open the file.
        try:
            output_file = open(filename, 'w')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        # Run the method compute_results if it was not already done.
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Writing some characteristics about the results on a file'
                  + '...')
        size = 60
        # Display the budget, the expenses, the percentage of budget spent
        # and the remaining budget.
        output_file.write('Expenses'.center(size-1, '='))
        output_file.write('\nBudget:'.ljust(size) + "{:,}".format(self.budget))
        output_file.write('\nAmount spent:'.ljust(size)
                          + "{:,.4f}".format(self.expenses))
        output_file.write('\nRemaining budget:'.ljust(size)
                          + "{:,.4f}".format(self.budget_remaining))
        output_file.write('\nPercentage of budget spent:'.ljust(size)
                          + "{:,.2%}".format(self.percent_budget))
        output_file.write('\n\n')
        # Display the initial total individual utility and the total individual
        # utility at optimal state.
        output_file.write('Individual Utility'.center(size-1, '='))
        output_file.write('\nInitial total individual utility:'.ljust(size)
                          + "{:,.4f}".format(self.init_utility))
        output_file.write('\nOptimal total individual utility:'.ljust(size)
                          + "{:,.4f}".format(self.optimal_utility))
        output_file.write('\n\n')
        # Display the initial energy consumption, the energy consumption at
        # optimal state and the energy gains.
        output_file.write('Energy Consumption'.center(size-1, '='))
        output_file.write('\nInitial energy consumption:'.ljust(size)
                          + "{:,.4f}".format(self.init_energy))
        output_file.write('\nEnergy gains:'.ljust(size)
                          + "{:,.4f}".format(self.total_energy_gains))
        output_file.write('\nOptimal energy consumption:'.ljust(size)
                          + "{:,.4f}".format(self.optimal_energy))
        output_file.write('\n\n')
        # Display information on the distance between the optimal state and the
        # last state (first best).
        output_file.write('Distance from First Best'.center(size-1, '='))
        output_file.write('\nMaximum budget necessary:'.ljust(size)
                          + "{:,.4f}".format(self.max_expenses))
        output_file.write(('\nActual budget in percentage of maximum budget '
                          + 'necessary:').ljust(size)
                          + "{:,.2%}".format(self.percent_max_expenses))
        output_file.write('\nMaximum energy gains:'.ljust(size)
                          + "{:,.4f}".format(self.max_total_energy_gains))
        output_file.write(('\nActual energy gains in percentage of maximum '
                          + 'energy gains:').ljust(size)
                          + "{:,.2%}".format(self.percent_total_energy_gains))
        output_file.write('\n\n')
        # Display the efficiency of the first jump and the last jump and the
        # efficiency of the optimal state.
        output_file.write('Efficiency'.center(size-1, '='))
        output_file.write('\nEfficiency of first jump:'.ljust(size)
                          + "{:,.4f}".format(self.first_efficiency))
        output_file.write('\nEfficiency of last jump:'.ljust(size)
                          + "{:,.4f}".format(self.last_efficiency))
        output_file.write('\nEfficiency of last state:'.ljust(size)
                          + "{:,.4f}".format(self.optimal_state_efficiency))
        output_file.write('\n\n')
        # Display the maximum, minimum and average amount of incentives and
        # energy gains.
        output_file.write('Jumps'.center(size-1, '='))
        output_file.write('\nLargest amount of incentives:'.ljust(size)
                          + "{:,.4f}".format(self.max_incentives))
        output_file.write('\nSmallest amount of incentives:'.ljust(size)
                          + "{:,.4f}".format(self.min_incentives))
        output_file.write('\nAverage amount of incentives:'.ljust(size)
                          + "{:,.4f}".format(self.average_incentives))
        output_file.write('\nLargest energy gains of a jump:'.ljust(size)
                          + "{:,.4f}".format(self.max_energy_gains))
        output_file.write('\nSmallest energy gains of a jump:'.ljust(size)
                          + "{:,.4f}".format(self.min_energy_gains))
        output_file.write('\nAverage energy gains of a jump:'.ljust(size)
                          + "{:,.4f}".format(self.average_energy_gains))
        output_file.write('\n\n')
        # Display the total number of jumps, the average number of jumps per
        # individual, the percentage of individuals that did not moved and the
        # percentage of individuals at their last alternative.
        output_file.write('Individuals and Jumps'.center(size-1, '='))
        output_file.write('\nTotal number of jumps:'.ljust(size)
                          + "{:,}".format(self.iteration))
        output_file.write(('\nAverage number of jumps per '
                          + 'individual:').ljust(size)
                          + "{:,.4f}".format(self.average_nb_jumps))
        output_file.write(('\nPercentage of individuals that did not '
                          + 'moved:').ljust(size)
                          + "{:,.2%}"
                          .format(self.percent_at_first_alternative))
        output_file.write(('\nPercentage of individuals at their last '
                          + 'alternative:').ljust(size)
                          + "{:,.2%}".format(self.percent_at_last_alternative))
        output_file.write('\n\n')
        # Display the interval of consumption energy at optimum.
        output_file.write('Bound'.center(size-1, '='))
        output_file.write('\nMinimum energy consumption at optimum:'
                          .ljust(size)
                          + "{:,.4f}".format(self.min_bound))
        output_file.write('\nMaximum energy consumption at optimum:'
                          .ljust(size)
                          + "{:,.4f}".format(self.max_bound))
        output_file.write('\nBound:'.ljust(size)
                          + "{:,.4f}".format(self.bound_size))
        output_file.write('\n\n')
        # Display computation times.
        output_file.write('Technical Information'.center(size-1, '='))
        if self.data.read_time is not None:
            output_file.write('\nTime to read the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.read_time))
        if self.data.generating_time is not None:
            output_file.write('\nTime to generate the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.generating_time))
        if self.data.sorting_time is not None:
            output_file.write('\nTime to sort the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.sorting_time))
        if self.data.pareto_removing_time is not None:
            output_file.write(('\nTime to remove Pareto-dominated'
                              + ' alternatives (s):').ljust(size)
                              + "{:,.4f}"
                              .format(self.data.pareto_removing_time))
        if self.data.output_data_time is not None:
            output_file.write('\nTime to output the data (s):'.ljust(size)
                              + "{:,.4f}".format(self.data.output_data_time))
        if self.data.output_characteristics_time is not None:
            output_file.write(('\nTime to output characteristics on the data '
                              + '(s):').ljust(size)
                              + "{:,.4f}".format(
                                  self.data.output_characteristics_time))
        if self.algorithm_running_time is not None:
            output_file.write('\nTime to run the algorithm (s):'.ljust(size)
                              + "{:,.4f}".format(self.algorithm_running_time))
        if self.computing_results_time is not None:
            output_file.write('\nTime to compute additional results (s):'
                              .ljust(size)
                              + "{:,.4f}".format(self.computing_results_time))
        if self.output_results_time is not None:
            output_file.write('\nTime to output the results (s):'.ljust(size)
                              + "{:,.4f}".format(self.output_results_time))
        # Store the time spent to output characteristics.
        self.output_characteristics_time = time.time() - init_time
        output_file.write(('\nTime to output characteristics on the results '
                          + '(s):').ljust(size)
                          + "{:,.4f}".format(self.output_characteristics_time))

    def output_results(self, filename, verbose=True):
        """Write a file with the results of the algorithm.

        The results include the alternative chosen by each individual at
        optimum and the amount of incentives to give to each one.

        :filename: string with the name of the file where the results are
        written
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        # Store the starting time.
        init_time = time.time()
        assert self.run_algorithm, \
            'The algorithm was not run.'
        # Try to open file filename and return FileNotFoundError if python is
        # unable to open the file.
        try:
            output_file = open(filename, 'w')
        except FileNotFoundError:
            print('Unable to open the file ' + str(filename))
            raise
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Writing the results of the algorithm on a file...')
        # The first row is the label of the three columns.
        output_file.write('Individual | Alternative | Incentives\n')
        # There is one row for each individual.
        for i in range(self.data.individuals):
            output_file.write(
                str(i+1).center(10)
                + ' |'
                + str(self.optimal_state[i]).center(13)
                + '| '
                + str(self.total_incentives[i]).center(10)
                + '\n'
            )
        # Store the time spent to output results.
        self.output_results_time = time.time() - init_time

    def plot_efficiency_curve(self, filename=None, dpi=200, show_title=True,
                              verbose=True):
        """Plot the efficiency curve with the algorithm results.

        The efficiency curve relates the expenses with the energy gains.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :title: title string to display on top of the graph
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the efficiency curve...')
        if show_title:
            title = 'Efficiency curve'
        else:
            title = None
        _plot_step_function(
                self.expenses_history,
                self.total_energy_gains_history,
                title=title,
                xlabel='Expenses',
                ylabel='Increase in social utility',
                filename=filename,
                dpi=dpi,
                left_lim=0,
        )

    def plot_efficiency_evolution(self, filename=None, dpi=200,
                                  title=None, xlabel='Iteration',
                                  ylabel='Efficiency', ylogscale=True,
                                  verbose=True):
        """Plot the evolution of the jump efficiency over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the jump efficiency...')
        _plot_scatter(
            x=self.iterations_history,
            y=self.efficiencies_history,
            marker='.',
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            regression=False,
            filename=filename,
            log_scale=ylogscale,
            dpi=dpi,
            left_lim=0,
        )

    def plot_expenses_curve(self, filename=None, dpi=200, show_title=True,
                            verbose=True):
        """Plot the expenses curve with the algorithm results.

        The expenses curve relates the expenses with the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the expenses curve...')
        if show_title:
            title = 'Expenses Curve'
        else:
            title = None
        _plot_step_function(
                self.iterations_history,
                self.expenses_history[:-1],
                title=title,
                xlabel='Iteration',
                ylabel='Expenses',
                filename=filename,
                dpi=dpi,
                left_lim=0,
        )

    def plot_incentives_evolution(self, filename=None, dpi=200,
                                  show_title=True, verbose=True):
        """Plot the evolution of the jump incentives over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the jump incentives...')
        if show_title:
            title = 'Evolution of the Incentives of the Jumps'
        else:
            title = None
        _plot_scatter(
                self.iterations_history,
                self.incentives_history,
                title,
                'Iteration',
                'Incentives',
                regression=False,
                filename=filename,
                dpi=dpi,
                left_lim=0,
        )

    def plot_energy_gains_curve(self, filename=None, dpi=200, show_title=True,
                                verbose=True):
        """Plot the energy gains curve with the algorithm results.

        The energy gains curve relates the energy gains with the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the energy gains curve...')
        if show_title:
            title = 'Energy gains Curve'
        else:
            title = None
        _plot_step_function(
            self.iterations_history,
            100*self.total_energy_gains_history[:-1]/self.total_energy_gains,
            title=title,
            xlabel='Iteration',
            ylabel='Increase in social utility (in % of total)',
            filename=filename,
            dpi=dpi,
            left_lim=0,
            top_lim=100,
            bottom_lim=0,
        )

    def plot_energy_gains_evolution(self, filename=None, dpi=200,
                                    show_title=True, verbose=True):
        """Plot the evolution of the jump energy gains over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the jump energy gains...')
        if show_title:
            title = 'Evolution of the Energy Gains of the Jumps'
        else:
            title = None
        _plot_scatter(
                self.iterations_history,
                self.energy_gains_history,
                title,
                'Iteration',
                'Increase in social utility',
                regression=False,
                filename=filename,
                dpi=dpi,
                left_lim=0,
        )

    def plot_bounds(self, filename=None, bounds=True, differences=True,
                    show_title=True, dpi=200, verbose=True):
        """Plot the lower and upper bounds of the total energy gains for each
        level of expenses. Also plot the difference between the lower and upper
        bounds.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :bounds: if True, plot the lower and upper bounds, default is True
        :difference: if True, plot the bound differences, default is True
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the bounds of the total energy gains...')
        if show_title:
            title = 'Energy gains bounds'
        else:
            title = None
        # Initiate the graph.
        fig, ax = plt.subplots()
        # The x-coordinates are the expenses history.
        x = self.expenses_history
        # The y-coordinates are the total energy gains history.
        y = self.total_energy_gains_history
        if bounds:
            # The upper bounds are an offset efficiency curve.
            ax.step(x, y, where='pre', label='Upper bounds', color=COLOR_1)
            # The lower bounds are the efficiency curve.
            ax.step(x, y, where='post', label='Lower bounds', color=COLOR_2)
        if differences:
            # Plot the bound differences.
            ax.step(x, self.bound_differences, 'b', where='post',
                    label='Bound differences', color=COLOR_3)
        # Add the title and the axis label.
        if title:
            ax.set_title(title)
        ax.set_xlabel('Expenses')
        ax.set_ylabel('Increase in social utility')
        # Display a legend.
        plt.legend()
        # Make room for the labels.
        plt.tight_layout()
        # Show the graph if no file is specified.
        if filename is None:
            plt.show()
        # Save the graph as a pdf file if a file is specified.
        else:
            plt.savefig(filename, dpi=dpi, format='pdf')
            plt.close()

    def plot_individuals_who_moved(self, filename=None, dpi=200,
                                   show_title=True, verbose=True):
        """Plot the evolution of the number of individuals who moved over the
        iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the number of individuals who '
                  + 'moved...')
        if show_title:
            title = 'Number of Individuals who Moved'
        else:
            title = None
        _plot_step_function(
                self.iterations_history,
                self.moved_history,
                title=title,
                xlabel='Iteration',
                ylabel='Number of individuals',
                filename=filename,
                dpi=dpi,
                left_lim=0,
        )

    def plot_individuals_at_first_best(self, filename=None, dpi=200,
                                       show_title=True, verbose=True):
        """Plot the evolution of the number of individuals at their last
        alternative over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print(('Plotting the evolution of the number of individuals at '
                  + 'first best...'))
        if show_title:
            title = 'Number of Individuals at First Best Alternative'
        else:
            title = None
        _plot_step_function(
                self.iterations_history,
                self.at_last_alternative_history,
                title=title,
                xlabel='Iteration',
                ylabel='Number of individuals',
                filename=filename,
                dpi=dpi,
                left_lim=0,
        )

    def plot_individuals_both(self, filename=None, dpi=200, show_title=True,
                              verbose=True):
        """Plot the evolution of the number of individuals who moved and the
        number of individuals at their last alternative over the iterations.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print(('Plotting the evolution of the status of individuals at '
                  + 'first best...'))
        # Initiate the graph.
        fig, ax = plt.subplots()
        # Plot the line.
        ax.step(
                self.iterations_history, self.at_last_alternative_history, 'b',
                where='post', color=COLOR_1,
                label='Individuals at socially optimal alternative'
        )
        ax.step(
                self.iterations_history, self.moved_history, 'b', where='post',
                color=COLOR_2, label='Individuals who received incentives'
        )
        # Add the title and the axis label.
        if show_title:
            ax.set_title('Evolution of the Status of the Individuals')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of individuals')
        # Change the limits for the x-axis and y-axis.
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        # Make room for the labels.
        plt.legend()
        plt.tight_layout()
        # Show the graph if no file is specified.
        if filename is None:
            plt.show()
        # Save the graph as a pdf file if a file is specified.
        else:
            plt.savefig(filename, dpi=dpi, format='pdf')
            plt.close()

    def plot_choice_share(self, filename=None, dpi=200, show_title=True,
                          verbose=True):
        """Plot the evolution of the share of each choice as expenses increase.

        This plot has a meaning only if the individuals share the same
        alternatives.

        :file: string with the name of the file where the graph is saved, if
        None show the graph but does not save it, default is None
        :labels: list of strings with the name of each choice, used as labels
        for the plot
        :verbose: if True, some information are displayed during the process,
        default is True

        """
        if not self.computed_results:
            self.compute_results(verbose=verbose)
        if verbose:
            print('Plotting the evolution of the share of each choice...')
        # Initiate the graph.
        fig, ax = plt.subplots()
        # Plot the line.
        ax.stackplot(
            self.expenses_history,
            self.choice_shares,
            labels=self.labels,
        )
        # Add the title and the axis label.
        if show_title:
            ax.set_title('Evolution of choice share')
        ax.set_xlabel('Expenses')
        ax.set_ylabel('Share')
        # Change the limits for the x-axis and y-axis.
        ax.set_xlim(0, self.expenses)
        ax.set_ylim(0, 1)
        # Make room for the labels.
        ax.legend()
        fig.tight_layout()
        # Show the graph if no file is specified.
        if filename is None:
            plt.show()
        # Save the graph as a pdf file if a file is specified.
        else:
            fig.savefig(filename, dpi=dpi, format='pdf')
            plt.close()


class Regression:

    """A Regression object stores information on the regression between two
    variables.
    """

    def __init__(self, x, y):
        """Initiate variables, perform a regression and create a legend. """
        self.x = x
        self.y = y
        self.func = _reg_func
        # Do a polynomial regression.
        self._polynomial_regression()
        # Create a legend.
        self._legend(4)

    def _polynomial_regression(self):
        """Compute the coefficients, the covariance matrix, the t-statistics
        and the significance (boolean) of a polynomial regression.
        """
        # Compute the coefficients and the covariance matrix of the regression.
        self.coefficients, self.covariance = curve_fit(
                self.func,
                self.x,
                self.y
        )
        # Compute the statistical significance of the coefficients (estimate
        # divided by its standard error).
        t_statistics = []
        for i in range(len(self.coefficients)):
            t = abs(self.coefficients[i]) / self.covariance[i, i]**(1/2)
            t_statistics.append(t)
        # The coefficients are statistically significant if the t-statistic is
        # greater than 1.96.
        t_statistics = np.array(t_statistics)
        self.significance = t_statistics > 1.96
        # Store the number of significant coefficients.
        self.nb_significant = sum(self.significance)
        # Store the significant coefficients.
        self.signi_coef = self.coefficients * self.significance

    def _r_squared(self):
        """Compute the R² of the regression.
        """
        # Compute the predicted values of y using only the significant
        # coefficients.
        y_hat = self.func(self.x, *self.signi_coef)
        # Compute the mean of y.
        y_bar = sum(self.y)/len(self.y)
        y_bars = np.repeat(y_bar, len(self.y))
        # Compute the sum of squared residuals and the total sum of squares.
        SSR = np.sum((y_hat - self.y)**2)
        SST = np.sum((y_bars - self.y)**2)
        # Compute the R².
        self.R2 = 1 - SSR / SST

    def _legend(self, r):
        """Create a string with information on the regression that can be
        displayed through the legend of a plot.

        :r: precision of round

        """
        if self.nb_significant == 0:
            self.legend = '$y$ = 0\n$R^2$=0'
        else:
            # Create a string with the equation of the regression line.
            equation = '$y$ = '
            first = True
            for i, coef in enumerate(self.signi_coef):
                # Only complete the string if the coef is significant.
                if coef != 0:
                    # If this is the first coef, add it to the string.
                    if first:
                        equation += str(round(coef, r))
                        first = False
                    # Else, add a plus or minus sign and then the absolute
                    # value of the coefficient.
                    else:
                        if coef > 0:
                            equation += '+ '
                        else:
                            equation += '- '
                        equation += str(abs(round(coef, r)))
                    equation += ' '
                    # Add the function of x.
                    if i <= 2:
                        # Case of x^2, x or 1.
                        equation += self._x_string(2-i)
                    if i == 3:
                        # Case of log(x).
                        equation += '$log(x)$'
                    equation += ' '
            # Compute the R².
            self._r_squared()
            # The first line of the string is the equation and the second line
            # is the R².
            self.legend = equation + '\n$R^2$ = ' + str(round(self.R2, r))

    def _x_string(self, deg):
        """Create a string with x and a specified exponent.

        For instance, if degree is d>1, return '$x^d$'.
        If the exponent is 1, return '$x$'.
        If the exponent is 0, return ''.

        :degree: degree associated, must be a int
        :returns: a string

        """
        if deg == 0:
            s = ''
        elif deg == 1:
            s = r'$x$'
        elif deg >= 2:
            s = r'$x^' + str(deg) + '$'
        return s


###############
#  Functions  #
###############

def convert_to_knapsack(alternatives_data):
    """Convert data on alternatives from the economic problem to the knapsack
    problem.
    """
    new_list = []
    for i, individual in enumerate(alternatives_data):
        # Python is magic...
        indiv = individual.copy()
        # Find utility and energy of default alternative.
        max_utility = np.max(indiv[:, 0])
        min_energy = np.min(indiv[indiv[:, 0] == max_utility, 1])
        # Convert utitity to weight and energy to value.
        indiv[:, 0] = max_utility - indiv[:, 0]
        indiv[:, 1] = min_energy - indiv[:, 1]
        new_list.append(indiv)
    return new_list


# 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross
# product.
# Returns a positive value, if OAB makes a counter-clockwise turn,
# negative for clockwise turn, and zero if the points are collinear.
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared
    # lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = [tuple(a) for a in points]
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple
    # times.
    if len(points) <= 1:
        return np.array(points)

    # Build lower hull.
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the
    # beginning of the other list.
    return np.array(lower[:-1] + upper[:-1])


def _known_custom_bar(max_length, text):
    """Create a progress bar of specified length.

    :max_length: max length of the progress bar, should be an integer
    :text: string printed in the middle of the progress bar, should be
    string
    :returns: ProgressBar object

    """
    blanks = 25 - len(text)
    bar = progressbar.ProgressBar(
            max_value=max_length,
            widgets=[
                text,
                '... ',
                blanks * ' ',
                progressbar.Timer(),
                ' ',
                progressbar.ETA(),
                ' ',
                progressbar.Bar(left='[', right=']', fill='-'),
                progressbar.Percentage()
            ])
    return bar


def _unknown_custom_bar(main_text, counter_text):
    """Create a progress bar of unknown length.

    :main_text: string printed in the middle of the progress bar, should be
    string
    :counter_text: string printed after the counter indicator, should be string
    :returns: ProgressBar object and counter integer

    """
    blanks = 25 - len(main_text)
    bar = progressbar.ProgressBar(
            max_value=progressbar.UnknownLength,
            widgets=[
                main_text,
                '... ',
                blanks * ' ',
                progressbar.Timer(),
                ' (',
                progressbar.Counter(),
                ' ',
                counter_text,
                ')',
                ' ',
                progressbar.AnimatedMarker()
            ])
    counter = 1
    return bar, counter


def _plot_step_function(x, y, title, xlabel, ylabel, filename=None,
                        left_lim=0, right_lim=None, bottom_lim=0, top_lim=None,
                        dpi=200):
    """Plot a step function.

    :x: list or numpy array with the x-coordinates
    :y: list or numpy array with the y-coordinates
    :title: title of the graph
    :xlabel: label of the x-axis
    :ylabel: label of the y-axis
    :file: string with the name of the file where the graph is saved, if
    None show the graph but does not save it, default is None

    """
    # Initiate the graph.
    fig, ax = plt.subplots()
    # Plot the line.
    ax.step(x, y, 'b', where='post', color=COLOR_1)
    # Add the title and the axis label.
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Do not show negative values on the y-axis if all values are positive.
    if ax.get_ylim()[0] < 0 and min(y) >= 0:
        ax.set_ylim(bottom=0)
    # Change the limits for the x-axis and y-axis.
    if left_lim is not None:
        ax.set_xlim(left=left_lim)
    if right_lim is not None:
        ax.set_xlim(right=right_lim)
    if bottom_lim is not None:
        ax.set_ylim(bottom=bottom_lim)
    if top_lim is not None:
        ax.set_ylim(top=top_lim)
    # Make room for the labels.
    plt.tight_layout()
    # Show the graph if no file is specified.
    if filename is None:
        plt.show()
    # Save the graph as a pdf file if a file is specified.
    else:
        plt.savefig(filename, dpi=dpi, format='pdf')
        plt.close()


def plot_multiple_step_function(title=None, xlabel='Expenses',
                                ylabel='Increase in social utility',
                                filename=None, left_lim=0, right_lim=None,
                                bottom_lim=0, top_lim=None, dpi=200):
    """Plot a graph with multiple step function from a list of
    AlgorithmResults.
    """
    # Initiate the graph.
    fig, ax = plt.subplots()
    # Run the algorithm on different data.
    for i, rho in enumerate([-.5, 0, .5]):
        data = Data()
        data.generate(correlation=-rho)
        res = data.run_algorithm()
        res.compute_results()
        # Plot the line.
        ax.step(res.expenses_history, res.total_energy_gains_history, 'b',
                where='post', label=r'$\rho={}$'.format(rho), color=COLOR[i])
    # Add the title and the axis label.
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Change the limits for the x-axis and y-axis.
    if left_lim is not None:
        ax.set_xlim(left=left_lim)
    if right_lim is not None:
        ax.set_xlim(right=right_lim)
    if bottom_lim is not None:
        ax.set_ylim(bottom=bottom_lim)
    if top_lim is not None:
        ax.set_ylim(top=top_lim)
    # Make room for the labels.
    plt.tight_layout()
    plt.legend()
    # Show the graph if no file is specified.
    if filename is None:
        plt.show()
    # Save the graph as a pdf file if a file is specified.
    else:
        plt.savefig(filename, dpi=dpi, format='pdf')
        plt.close()


def _plot_scatter(x, y, title, xlabel, ylabel, marker='.', regression=False,
                  filename=None, left_lim=0, right_lim=None, bottom_lim=0,
                  top_lim=None, dpi=200, fraction=.8, log_scale=False):
    """Plot a scatter.

    :x: list or numpy array with the x-coordinates
    :y: list or numpy array with the y-coordinates
    :title: title of the graph
    :xlabel: label of the x-axis
    :ylabel: label of the y-axis
    :regression: if true, perform a regression and display the regression line
    and a legend
    :file: string with the name of the file where the graph is saved, if
    None show the graph but does not save it, default is None
    :log_scale: if true, the y axis is in log scale
    """
    # Initiate the graph.
    fig, ax = plt.subplots(figsize=set_size(fraction=fraction))
    # Plot the scatter.
    ax.scatter(x, y, marker=marker, s=5, color=COLOR_1)
    # Perform a regression if necessary.
    if regression:
        reg = Regression(x, y)
        xs = np.linspace(*ax.get_xlim(), 1000)
        ys = reg.func(xs, *reg.signi_coef)
        ax.plot(xs, ys, color=COLOR_2, label=reg.legend)
        ax.legend()
    # Do not show negative values on the y-axis if all values are positive.
    if ax.get_ylim()[0] < 0 and min(y) >= 0:
        ax.set_ylim(bottom=0)
    # Change the y axis to log scale.
    if log_scale:
        ylabel += ' (log scale)'
        ax.set_yscale('symlog', basey=10)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
    # Add the title and the axis label.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Change the limits for the x-axis and y-axis.
    if left_lim is not None:
        ax.set_xlim(left=left_lim)
    if right_lim is not None:
        ax.set_xlim(right=right_lim)
    if bottom_lim is not None:
        ax.set_ylim(bottom=bottom_lim)
    if top_lim is not None:
        ax.set_ylim(top=top_lim)
    # Show a grid.
    ax.grid()
    # Show the graph if no file is specified.
    if filename is None:
        plt.show()
    # Save the graph as a pdf file if a file is specified.
    else:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()


def _reg_func(x, a, b, c, d):
    """Define a function used to compute the regressions."""
    return (a * x**2) + (b * x) + c + (d * np.log(x))


def _simulation(budget=np.infty, rem_eff=True, small_jumps=False, verbose=True,
                **kwargs):
    """Generate random data and run the algorithm.

    To specify the parameters for the generation process, use the same syntax
    as for the method Data.generate().

    :budget: budget used to run the algorithm, by default budget is infinite
    :rem_eff: if True, the efficiency dominated alternatives
    are removed before the algorithm is run
    :small_jumps: if True, use the "small jumps" technique
    :verbose: if True, display progress bars and some information
    :returns: an AlgorithmResults object with the results of the algorithm run

    """
    # Create a Data object.
    data = Data()
    # Generate random data.
    data.generate(verbose=verbose, **kwargs)
    if rem_eff:
        # Remove the efficiency dominated alternatives and run the lite version
        # of the algorithm.
        data.remove_efficiency_dominated(verbose=verbose)
        results = data.run_lite_algorithm(budget=budget, verbose=verbose)
    else:
        # Run the normal version of the algorithm.
        results = data.run_algorithm(
                budget=budget, small_jumps=small_jumps, verbose=verbose
        )
    return results


def _run_algorithm(simulation=False, filename=None, budget=np.infty,
                   remove_efficiency_dominated=True, directory='files',
                   dpi=200, show_title=True, delimiter=',', comment='#',
                   small_jumps=False, verbose=True, **kwargs):
    """Run the algorithm and generate files and graphs.

    The algorithm can be run with generated data or with imported data.

    :simulation: boolean indicated whether the data must be generated or
    imported
    :filename: string with the name of the file containing the data
    :budget: budget used to run the algorithm, by default budget is infinite
    :remove_efficiency_dominated: if True, the efficiency dominated
    alternatives are removed before the algorithm is run
    :directory: directory where the files are stored, must be a string, default
    is 'files'
    :dpi: dpi used for generating the file
    :show_title: if True, add a title on top of the graphs
    :delimiter: the character used to separated the utility and the energy
    consumption of the alternatives, default is comma
    :comment: line starting with this string are not read, should be a
    string, default is #
    :small_jumps: if True, use the "small jumps" technique
    :verbose: if True, display progress bars and some information

    """
    # Store the starting time.
    init_time = time.time()
    # Create the directory used to store the files.
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    if simulation:
        # Run the simulation.
        results = _simulation(
                budget=budget,
                rem_eff=remove_efficiency_dominated,
                small_jumps=small_jumps,
                verbose=verbose,
                **kwargs
        )
    else:
        # Import the data.
        data = Data()
        data.read(filename, delimiter=delimiter, comment=comment,
                  verbose=verbose)
        # Run the algorithm.
        results = data.run_algorithm(
                budget=budget, small_jumps=small_jumps, verbose=verbose
        )
    # Generate the files and the graphs.
    results.data.output_data(filename=directory+'/data.txt', verbose=verbose)
    results.data.output_characteristics(
            filename=directory+'/data_characteristics.txt',
            verbose=verbose
            )
    results.output_results(
            filename=directory+'/results.txt',
            verbose=verbose
            )
    results.output_characteristics(
            filename=directory+'/results_characteristics.txt',
            verbose=verbose
            )
    results.plot_efficiency_curve(
            filename=directory+'/efficiency_curve.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_efficiency_evolution(
            filename=directory+'/efficiency_evolution.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_expenses_curve(
            filename=directory+'/expenses_curve.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_incentives_evolution(
            filename=directory+'/incentives_evolution.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_energy_gains_curve(
            filename=directory+'/energy_gains_curve.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_energy_gains_evolution(
            filename=directory+'/energy_gains_evolution.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_bounds(
            filename=directory+'/bounds.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_individuals_who_moved(
            filename=directory+'/individuals_who_moved.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_individuals_at_first_best(
            filename=directory+'/individuals_at_first_best.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    results.plot_individuals_both(
            filename=directory+'/individuals_both.pdf',
            verbose=verbose,
            dpi=dpi,
            show_title=show_title,
            )
    # Store the total time to run the simulation.
    total_time = time.time() - init_time
    if verbose:
        print('Finished! (Elapsed Time: '
              + str(round(total_time, 2))
              + 's)')


def run_simulation(budget=np.infty, directory='files', dpi=200,
                   small_jumps=False, verbose=True, **kwargs):
    """Create files and graphs while generating random data and running the
    algorithm.

    To specify the parameters for the generation process, use the same syntax
    as for the method Data.generate().
    The generated files are the data, data characteristics, the results and
    results characteristics.
    The generated graphs are efficiency curve, efficiency evolution, incentives
    evolution, energy gains evolution, bounds, individuals who moved and
    individuals at first best.

    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: directory where the files are stored, must be a string, default
    is 'files'
    :dpi: dpi used for generating the file
    :small_jumps: if True, use the "small jumps" technique
    :verbose: if True, display progress bars and some information

    """
    _run_algorithm(simulation=True, budget=budget, directory=directory,
                   dpi=dpi, small_jumps=small_jumps, verbose=verbose, **kwargs)


def run_from_file(filename, budget=np.infty, directory='files', delimiter=',',
                  comment='#', small_jumps=False, verbose=True, **kwargs):
    """Read data from a file and run the algorithm.

    The generated files are the data, data characteristics, the results and
    results characteristics.
    The generated graphs are efficiency curve, efficiency evolution, incentives
    evolution, energy gains evolution, bounds, individuals who moved and
    individuals at first best.

    :filename: string with the name of the file containing the data
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: directory where the files are stored, must be a string, default
    is 'files'
    :delimiter: the character used to separated the utility and the energy
    consumption of the alternatives, default is comma
    :comment: line starting with this string are not read, should be a
    string, default is #
    :small_jumps: if True, use the "small jumps" technique
    :verbose: if True, display progress bars and some information

    """
    _run_algorithm(filename=filename, budget=budget, directory=directory,
                   delimiter=delimiter, comment=comment,
                   small_jumps=small_jumps, verbose=verbose, **kwargs)


def _complexity(varying_parameter, string, start, stop, step, budget=np.infty,
                directory='complexity', remove_pareto=True, verbose=True,
                **kwargs):
    """Run multiple simulations with a parameter varying and compute time
    complexity of the algorithm.

    :varying_parameter: string specifying the parameter which varies across
    simulations, possible values are 'individuals', 'alternatives' and 'budget'
    :string: string with the name of the varying parameter, used to label the
    graphs
    :start: start value for the interval of number of individuals
    :stop: end value for the interval of number of individuals, this value is
    not include in the interval
    :step: spacing between values in the interval
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: string specifying the directory where the files are stored
    :remove_pareto: boolean indicating wether to remove the Pareto-dominated
    alternatives before running the algorithm
    :verbose: if True, a progress bar and some information are displayed during

    """
    # Check that varying_parameter is well specified.
    assert varying_parameter in ['individuals', 'alternatives', 'budget'], \
        'The varying parameter is not well specified'
    # Create the directory used to store the files.
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    # Compute the interval of values for the number of individuals.
    X = np.arange(start, stop, step, dtype=int)
    if verbose:
        # Print a progress bar of duration the number of simulations.
        bar = _known_custom_bar(len(X), 'Running simulations')
    # Generate empty lists to store the computing times.
    generating_times = []
    pareto_removing_times = []
    running_times = []
    # Run a simulation for each value in the interval and store relevant
    # results.
    for i, x in enumerate(X):
        if verbose:
            bar.update(i)
        data = Data()
        time0 = time.time()
        # Generate the data.
        if varying_parameter == 'individuals':
            data.generate(individuals=x, verbose=False, **kwargs)
        elif varying_parameter == 'alternatives':
            data.generate(mean_nb_alternatives=x, verbose=False, **kwargs)
        elif varying_parameter == 'budget':
            data.generate(verbose=False, **kwargs)
        time1 = time.time()
        generating_times.append(time1 - time0)
        if remove_pareto:
            # Remove the Pareto dominated alternatives.
            data.remove_pareto_dominated(verbose=False)
        time2 = time.time()
        pareto_removing_times.append(time2 - time1)
        # Run the algorithm.
        if varying_parameter == 'budget':
            data.run_lite_algorithm(budget=x, verbose=False)
        else:
            data.run_lite_algorithm(budget=budget, verbose=False)
        time3 = time.time()
        running_times.append(time3 - time2)
    bar.finish()
    # Plot the time in ms.
    generating_times = np.array(generating_times)*1000
    pareto_removing_times = np.array(pareto_removing_times)*1000
    running_times = np.array(running_times)*1000
    # Plot graphs showing time complexity.
    _plot_scatter(
            X,
            generating_times,
            '',
            string,
            'Generating Time (ms)',
            filename=directory+'/generating_time.pdf',
            left_lim=start-step,
            bottom_lim=0
            )
    if remove_pareto:
        _plot_scatter(
                X,
                pareto_removing_times,
                '',
                string,
                'Time to Remove Pareto-Dominated Alternatives (ms)',
                filename=directory+'/removing_pareto_dominated_times.pdf',
                left_lim=start-step,
                bottom_lim=0
                )
    _plot_scatter(
            X,
            running_times,
            '',
            string,
            'Running Time (ms)',
            filename=directory+'/running_time.pdf',
            left_lim=start-step,
            bottom_lim=0
            )
    if verbose:
        print('Successfully ran ' + str(len(X)) + ' simulations.')
    return running_times


def complexity_individuals(start, stop, step, budget=np.infty,
                           directory='complexity_individuals',
                           remove_pareto=False, verbose=True, **kwargs):
    """Run multiple simulations with a varying number of individuals and
    compute time complexity of the algorithm.

    To specify the parameters for the generation process, use the same syntax
    as for the method Data.generate().

    :start: start value for the interval of number of individuals
    :stop: end value for the interval of number of individuals, this value is
    not include in the interval
    :step: spacing between values in the interval
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: string specifying the directory where the files are stored
    :remove_pareto: boolean indicating wether to remove the Pareto-dominated
    alternatives before running the algorithm
    :verbose: if True, a progress bar and some information are displayed during
    the process, default is True

    """
    string = 'Number of Individuals'
    running_times = _complexity('individuals', string, start, stop, step,
                                budget=budget, directory=directory,
                                remove_pareto=remove_pareto, verbose=verbose,
                                **kwargs)
    return running_times


def complexity_alternatives(start, stop, step, budget=np.infty,
                            directory='complexity_alternatives',
                            remove_pareto=False, verbose=True, **kwargs):
    """Run multiple simulations with a varying average number of alternatives
    and compute time complexity of the algorithm.

    To specify the parameters for the generation process, use the same syntax
    as for the method Data.generate().

    :start: start value for the interval of average number of alternatives
    :stop: end value for the interval of average number of alternatives, this
    value is not include in the interval
    :step: spacing between values in the interval
    :budget: budget used to run the algorithm, by default budget is infinite
    :directory: string specifying the directory where the files are stored
    :remove_pareto: boolean indicating wether to remove the Pareto-dominated
    alternatives before running the algorithm
    :verbose: if True, a progress bar and some information are displayed during
    the process, default is True

    """
    string = 'Average Number of Alternatives'
    _complexity('alternatives', string, start, stop, step, budget=budget,
                directory=directory, remove_pareto=remove_pareto,
                verbose=verbose, **kwargs)


def complexity_budget(start, stop, step, directory='complexity_budget',
                      remove_pareto=True, verbose=True, **kwargs):
    """Run multiple simulations with a varying budget and compute time
    complexity of the algorithm.

    To specify the parameters for the generation process, use the same syntax
    as for the method Data.generate().

    :start: start value for the interval of budget
    :stop: end value for the interval of budget, this value is not include in
    the interval
    :step: spacing between values in the interval
    :directory: string specifying the directory where the files are stored
    :remove_pareto: boolean indicating wether to remove the Pareto-dominated
    alternatives before running the algorithm
    :verbose: if True, a progress bar and some information are displayed during
    the process, default is True

    """
    string = 'Budget'
    _complexity('budget', string, start, stop, step, directory=directory,
                remove_pareto=remove_pareto, verbose=verbose, **kwargs)


def distance_optimum(individuals=10, filename='distance_optimum', file=None,
                     bounds=False, title='Distance to the Optimum',
                     xlabel='Budget', ylabel='Social welfare',
                     verbose=True, dpi=200, left_lim=0, **kwargs):
    """Run the algorithm and find the optimum on a small sample, then draw a
    graph showing the distance of the algorithm from the optimum.

    To specify the parameters for the generation process, use the same syntax
    as for the method Data.generate().

    :individuals: number of individuals in the generated data
    :filename: string with the name of the file where the graph is saved, if
    None show the graph but does not save it, default is None
    :verbose: if True, a progress bar and some information are displayed during
    the process, default is True

    """
    data = Data()
    if file:
        data.read(file)
    else:
        data.generate(individuals=individuals, verbose=verbose, **kwargs)
    optimums = data.find_optimum(verbose=verbose)
    alg_res = data.run_algorithm(verbose=verbose)
    alg_res.compute_results(verbose=verbose)
    if verbose:
        print('Plotting the graph...')
    # Initiate the graph.
    fig, ax = plt.subplots(figsize=set_size(fraction=.8))
    # Plot the optimums.
    ax.step(
        optimums['cost'], optimums['energy'], where='post', alpha=.9,
        label=r'Exact social welfare curve $\mathcal{C}^*_Q$', color=COLOR_3
    )
    # The x-coordinates are the expenses history.
    x = alg_res.expenses_history
    # The y-coordinates are the total energy gains history.
    y = -alg_res.energy_history
    if bounds:
        ax.step(
            x, y, where='post', alpha=.9,
            label=r'Approximate social welfare curve $\mathcal{C}_Q$',
            color=COLOR_2, linestyle='dashed',
        )
        ax.plot(
            x, y, label=r'Upper bound of $\mathcal{C}^*_Q$', alpha=.9,
            color=COLOR_1, linestyle='dotted'
        )
        ax.scatter(
            x, y, color=COLOR_2, marker='*', s=50, zorder=5, alpha=.9,
            label=r'Point $\big(Y^{[k]}, B^*(Y^{[k]})\big)$ of $\mathcal{C}_Q$',
        )
    else:
        # Plot the algorithm results (efficiency curve).
        ax.step(
            x, y, where='post', label='Approximate social welfare curve',
            color=COLOR_2, linestyle='dashed', marker='o', markersize=5,
        )
    if left_lim is not None:
        ax.set_xlim(left=left_lim)
    # Add the title and the axis label.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Show a grid.
    ax.grid()
    # Display a legend.
    plt.legend()
    # Show the graph if no filename is specified.
    if filename is None:
        plt.show()
    # Save the graph if a filename is specified.
    else:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()


def running_time_complexity_individuals(start, stop, step):
    x = np.arange(start, stop, step)
    total = np.sum(0.0028 * x**2)/1000
    return total


def running_time_complexity_alternatives(start, stop, step):
    x = np.arange(start, stop, step)
    total = np.sum(0.3302 * x**2 - 39.0274 * x + 1287.7034 * np.log(x))/1000
    return total


def imperfect_info(data, budget, imperfect_budget, offset=0):
    results = data.run_lite_algorithm(budget=imperfect_budget, verbose=False)
    results.compute_results(verbose=False)
    data.add_epsilons(scale=.1)
    incentives = np.array(results.total_incentives) + offset
    imperfect_choices = data.get_choices(
        incentives=incentives,
        init_state=results.init_state,
        last_state=results.optimal_state,
    )
    imperfect_state = results.optimal_state * imperfect_choices \
        + results.init_state * ~imperfect_choices
    imperfect_budget = np.sum(incentives*imperfect_choices)
    perfect_results = data.run_lite_algorithm(budget=budget, verbose=False)
    init_energy = data._total_energy(perfect_results.init_state)
    imperfect_energy = data._total_energy(imperfect_state)
    imperfect_energy_gains = init_energy - imperfect_energy
    perfect_energy_gains = perfect_results.total_energy_gains
    perfect_budget = perfect_results.expenses
    # Return.
    d = {
        'imperfect_energy_gains': imperfect_energy_gains,
        'perfect_energy_gains': perfect_energy_gains,
        'imperfect_budget': imperfect_budget,
        'perfect_budget': perfect_budget,
    }
    return d


def set_size(width=470, ratio='golden', fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if ratio == 'golden':
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
