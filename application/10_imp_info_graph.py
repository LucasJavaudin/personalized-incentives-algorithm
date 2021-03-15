"""Script to plot graphs for the imperfect information section.

Author: Lucas Javaudin
E-mail: lucas.javaudin@ens-paris-saclay.fr
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../python')

from algorithm import COLOR_1, COLOR_2, COLOR_3, set_size

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

# Compute probability of acceptance, as a function of the difference in
# deterministic utility (y_hat), when the incentive amount (y) corresponds to
# the expected difference in deterministic utility.
x_min, x_max = -5, 5
y_hat = np.linspace(x_min, x_max, 200)
y = (1+np.exp(-y_hat)) * np.log(1+np.exp(y_hat))
prob = (1-np.exp(-y)) / (1+np.exp(y_hat-y))
# Plot the incentive amount and the probability.
fig, ax1 = plt.subplots(figsize=set_size(fraction=.8))
ax2 = ax1.twinx()
ax1.plot(y_hat, y, color=COLOR_1, label='Incentive amount', linestyle='dashed')
ax2.plot(y_hat, prob, color=COLOR_2, label='Probability of acceptance')
ax1.set_xlabel(r'Difference in deterministic utility ($\hat{y}_{i, j}$)')
ax1.set_ylabel(r'Incentive amount ($y_{i, j}$)')
ax2.set_ylabel(r'Probability of acceptance ($\pi_{i, j}(y_{i, j}))$')
ax1.set_xlim(x_min, x_max)
ax2.set_ylim(0, 1)
ax1.legend()
ax2.legend()
ax1.grid()
ax2.grid()
fig.tight_layout()
fig.savefig('output/acceptance_probability.pdf')
