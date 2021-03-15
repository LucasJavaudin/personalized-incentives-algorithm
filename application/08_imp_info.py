"""Script to run the MCKP algorithm with imperfect information.

The `data/` directory must contain the algorithm input (script 04).

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.append('../python')

from algorithm import Data, COLOR_1, COLOR_2, COLOR_3, set_size

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

# Import algorithm input.
print('Importing data')
input_df = pd.read_csv(
    'data/algorithm_input_rhone.csv',
    usecols=['obs_id', 'alt_id', 'minus_incentive', 'true_utility', 'kg_co2',
             'choice'],
)
df = pd.read_csv('data/long_data_rhone.csv')
df = df.set_index(['obs_id', 'alt_id'])

# Create MCKP Data, using expected incentives.
print('Reading data')
data = Data()
data.read_dataframe(input_df, indiv_col='obs_id',
                    utility_col='minus_incentive', energy_col='kg_co2',
                    label_col='alt_id', verbose=False)

print('Running algorithm')
budget = np.inf
results = data.run_lite_algorithm(budget=budget, verbose=False)
print('Computing results')
results.compute_results()

jumps_history = list()
id_to_case = input_df['obs_id'].unique()
for i, prev_j, next_j in results.jumps_history:
    case = id_to_case[i]
    prev_alt = data.alternative_labels[i][prev_j]
    next_alt = data.alternative_labels[i][next_j]
    jumps_history.append([case, prev_alt, next_alt])

input_df = input_df.set_index(['obs_id', 'alt_id'])
budget = 1800
expenses = 0
energy_gains = 0
nb_jumps = 0
nb_accepted = 0
utility_inc = 0
utilities = input_df.loc[input_df['choice'], 'true_utility']
utilities = utilities.droplevel('alt_id').to_dict()
emissions = input_df.loc[input_df['choice'], 'kg_co2']
emissions = emissions.droplevel('alt_id').to_dict()
incentives = dict()
true_incentives = list()
for jump, incentive in zip(jumps_history, results.incentives_history):
    nb_jumps += 1
    if jump[0] in incentives:
        current_incentive = incentives[jump[0]]
    else:
        current_incentive = 0
    old_utility = utilities[jump[0]]
    new_utility = input_df.loc[(jump[0], jump[2]), 'true_utility']
    true_incentives.append(old_utility - new_utility)
    new_utility += incentive + current_incentive
    if new_utility >= old_utility:
        # The incentive is accepted.
        if expenses + incentive > budget:  # Stop when budget is depleted.
        # if expenses + incentive - utility_inc > budget:
        # Stop when total surplus decrease exceeds budget.
            nb_jumps -= 1
            break
        else:
            utilities[jump[0]] = new_utility
            utility_inc += new_utility - old_utility
            nb_accepted += 1
            expenses += incentive
            old_energy = emissions[jump[0]]
            new_energy = input_df.loc[(jump[0], jump[2]), 'kg_co2']
            emissions[jump[0]] = new_energy
            energy_gain = old_energy - new_energy
            energy_gains += energy_gain
            incentives[jump[0]] = incentive + current_incentive
print('Number of incentives proposed: {}'.format(nb_jumps))
print('Number of incentives accepted: {}'.format(nb_accepted))
print('Rate of acceptance: {:%}'.format(nb_accepted/nb_jumps))
print('Expenses: {}'.format(expenses))
print('Decrease in CO2 emissions: {}'.format(energy_gains))
print('Increase in individual utility: {}'.format(utility_inc))
print('Total surplus variation: {}'.format(utility_inc-expenses))

fig, ax = plt.subplots()
xs = results.incentives_history[:nb_jumps+1]
ys = true_incentives
c = ['green' if x > y else 'red' for x, y in zip(xs, ys)]
ax.scatter(xs, ys, s=1, c=c, alpha=.3)
M = min(np.max(xs), np.max(ys))
ax.plot([0, M], [0, M], color='black')
ax.set_xlabel('Incentive given')
ax.set_ylabel('Minimum incentive needed')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig('output/imperfect_incentives.pdf')
