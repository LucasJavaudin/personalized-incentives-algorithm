Personalized-Incentives Algorithm
=================================

Algorithm to compute optimal allocation of personalized incentives in a discrete-choice framework.

How-To Run the Algorithm
------------------------

There is currently no GUI implemented to run the algorithm.
You can run the algorithm by typing the following command line in a terminal.
> python -i /path/to/file/algorithm.py

Then, you can type commands in the python interpreter.

### Command _run_from_file()_

This command import data from a file and run the algorithm.

#### Parameters

You must specify the name of the file containing the data as a string (e.g. 'data.txt').
Python look for the file in the current working directory.
Alternatively, you can specify the complete path to the file.

Optionally, you can specify the budget used to run the algorithm with the parameter _budget_.
By default, the budget is infinite.

#### Input file

The file containing the data must have two lines for each individual.
The two first lines describe the alternatives of the first individual, the two subsequent lines describe the alternatives of the second individual, etc.

The first line of each individual contains the utility of the alternatives and the second line contains the energy consumption of the alternatives.
The two lines must contain the same number of values.
The values are separated by a comma (you can change the delimiter character with the parameter _delimiter_).

Lines starting with '#' are not read by Python (you can change the comment character with the parameter _comment_).

The file _sample_data.txt_ in the directory _python_ is an example of input file.

#### Output

If no error occurred, 4 text files and 6 graphs are generated and saved in directory _files/_.
If you run the command a second time, the previous files are deleted.
You can store the files in a different directory using the parameter _directory_ (see example below).
- _data.txt_: file with the generated data (utility and energy gains for all alternatives)
- _data_characteristics.txt: file with some characteristics on the data (number of individual, total energy gains possible, etc.)
- _results.txt: file with the results of the algorithm (final choice and amount of incentives for all individuals)
- _results_characteristics.txt_: file with some characteristics on the results (expenses, energy gains, number of iterations, etc.)
- _efficiency_curve.png_: graph plotting total energy gains against expenses
- _efficiency_evolution.png_: graph plotting efficiency of the jumps against iterations
- _incentives_evolution.png_: graph plotting amount of incentives of the jumps against iterations
- _energy_gains.png_: graph plotting energy gains of the jumps against iterations
- _bounds.png: graph with lower and upper bounds for total energy gains
- _individuals_who_moved.png_: graph plotting number of individuals who received incentives against iterations
- _individuals_at_first_best.png_: graph plotting number of individuals at first best alternative (alternative with the most energy gains) against iterations

#### Example

The following command imports data from the file _sample_data.txt_, runs the algorithm with a budget of 10000 and stores the results in the directory _/results_.
> run_from_file('sample_data.txt', budget=10000, directory='results')

The following command imports data from the file _input.txt_ with semi-colons as delimiter and with '*' as comment character, and then run the algorithm.
> run_from_file('input.txt', delimiter=';', comment='*')

### Command _run_simulation()_

This command generate random data, sort the data, remove the Pareto-dominated alternatives and run the algorithm.

#### Parameters

You can change the value of the parameters used in the generating process:
- _individuals_ (default: 1000): number of individuals generated
- _mean_nb_alternatives_ (default: 10): average number of alternatives per individual
- _use_poisson_ (default: True): if True, the number of alternatives is drawn from a Poisson distribution, else the number of alternatives is fixed to mean_nb_alternatives
- _use_gumbel_ (default: False): if True, stochastic utility is generated from the Gumbel distribution, else the Logistic distribution is used
- _random_utility_parameter_ (default: 10): parameter of the distribution used to generate stochastic utility (Gumbel or Logistic)
- _utility_mean_ (default: 1): mean parameter for the log-normal distribution used to generate the utility of the alternatives
- _utility_sd_ (default: 1): standard-deviation parameter for the log-normal distribution used to generate the utility of the alternatives
- _alpha_ (default: 1): energy consumption of an alternative is defined by _alpha * (U ^ gamma) + beta_ where _U_ is the utility of the alternative
- _beta_ (default: 0): see alpha
- _gamma_ (default: 1): see alpha

You can also change the budget used to run the algorithm with the parameter _budget_.
By default, the budget is infinite (the algorithm run until all individuals are at the alternative with the most energy gains).

#### Output

If no error occurred, 4 text files and 6 graphs are generated and saved in directory _files/_.
If you run the command a second time, the previous files are deleted.
You can store the files in a different directory using the parameter _directory_ (see example below).
- _data.txt_: file with the generated data (utility and energy gains for all alternatives)
- _data_characteristics.txt: file with some characteristics on the data (number of individual, total energy gains possible, etc.)
- _results.txt: file with the results of the algorithm (final choice and amount of incentives for all individuals)
- _results_characteristics.txt_: file with some characteristics on the results (expenses, energy gains, number of iterations, etc.)
- _efficiency_curve.png_: graph plotting total energy gains against expenses
- _efficiency_evolution.png_: graph plotting efficiency of the jumps against iterations
- _incentives_evolution.png_: graph plotting amount of incentives of the jumps against iterations
- _energy_gains.png_: graph plotting energy gains of the jumps against iterations
- _bounds.png: graph with lower and upper bounds for total energy gains
- _individuals_who_moved.png_: graph plotting number of individuals who received incentives against iterations
- _individuals_at_first_best.png_: graph plotting number of individuals at first best alternative (alternative with the most energy gains) against iterations

#### Example

The following command generates data with 500 individuals and 20 alternatives per individual on average, then runs the algorithm with a budget of 10000 and stores the results in the directory _results/_:
> run_simulation(budget=10000, individuals=500, mean_nb_alternatives=20, directory='results')

### Commands _complexity_individuals()_, _complexity_alternatives()_ and _complexity_budget()_

These commands run multiple simulations and plot graphs showing time complexity.
With _complexity_individuals()_, the number of individuals varies across simulations; with _complexity_alternatives()_, the average number of alternatives varies across simulations; with _complexity_budget()_, the budget varies across simulations.

#### Parameters

All these commands have 3 mandatory parameters to specify the interval used for the varying number of individuals (for _complexity_individuals()_), for the varying average number of alternatives (for _complexity_alternatives()_) or for the varying budget (for _complexity_budget()_).
- _start_: start value of the interval
- _stop_: end value of the interval
- _step_: spacing between values in the interval

The other parameters used in the simulations are set to their default value (the same value than with _run_simulation()_).
You can change the value of these parameters using the same syntax (see example below).

Additionally, you can change the budget used to run the algorithm with the parameter _budget_.
By default, the budget is infinite.
Note that this parameter does not work with _complexity_budget()_.

#### Output

All these commands generate 3 graphs. By defaults, the graphs are stored in the directory _complexity_individuals/_, _complexity_alternatives/_ or _complexity_budget/_.
You can store the files in a different directory using the parameter _directory_ (see example below).
- _generating_times.png_: graph showing time complexity for the time spent generating the data
- _cleaning_times.png_: graph showing time complexity for the time spent cleaning the data (sorting and removing of Pareto-dominated alternatives)
- _running_time.png_: graph showing time complexity for the time spent running the algorithm

#### Examples

To run 90 simulations with the number of individuals varying from 100 to 990 (step of 10) and with the graphs stored in the directory _complexity_results/_, use:
> complexity_individuals(100, 1000, 10, directory='complexity_results')

To run 45 simulations with the average number of alternatives varying from 5 to 49 (step of 1), with 500 individuals and with a budget of 10000, use:
> complexity_alternatives(5, 50, 1, individuals=500, budget=10000)
