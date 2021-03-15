"""Script to solve the MCKP problem using OR-Tools.

The `data/` directory must contain the algorithm input (script 04).

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import sys
import time

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

sys.path.append('../python')

from algorithm import Data

# Import algorithm input.
print('Importing data')
input_df = pd.read_csv(
    'data/algorithm_input_rhone.csv',
    usecols=['obs_id', 'alt_id', 'true_utility', 'kg_co2', 'choice'],
)
df = pd.read_csv('data/long_data_rhone.csv')
od_matrix = pd.read_csv('data/od_matrix_rhone.csv')
df = df.merge(od_matrix[['home', 'work', 'geodesic_distance']],
              on=['home', 'work'], how='left')
df = df.set_index(['obs_id', 'alt_id'])

# Create MCKP Data.
print('Reading data')
data = Data()
data.read_dataframe(input_df, indiv_col='obs_id', utility_col='true_utility',
                    energy_col='kg_co2', label_col='alt_id',
                    verbose=False)
data.sort()
print('Number of individuals: {}'.format(data.individuals))
print('Number of alternatives: {}'.format(
    np.sum(data.alternatives_per_individual)))

# Convert utility to incentive amount.
for values in data.list:
    values[:, 0] = values[0, 0] - values[:, 0]

budget = 1700

# Create the mip solver with the SCIP backend.
print('Initializing solver')
solver = pywraplp.Solver.CreateSolver('SCIP')

# Variables
# x[i, j] = 1 if individual i chooses alternative j.
x = {}
for i, J in enumerate(data.alternatives_per_individual):
    for j in range(J):
        x[(i, j)] = solver.IntVar(0, 1, 'x_{}_{}'.format(i, j))

# Constraints
# Each individual can choose most one alternative.
for i, J in enumerate(data.alternatives_per_individual):
    solver.Add(sum(x[i, j] for j in range(J)) == 1)
# Total expenses cannot exceed the budget.
solver.Add(sum(x[(i, j)] * data.list[i][j, 0]
               for i, J in enumerate(data.alternatives_per_individual)
               for j in range(J))
           <= budget)

# Objective
objective = solver.Objective()

for i, J in enumerate(data.alternatives_per_individual):
    for j in range(J):
        objective.SetCoefficient(x[(i, j)], -data.list[i][j, 1])
objective.SetMaximization()

print('Starting solver')
t0 = time.time()
status = solver.Solve()
t1 = time.time()
print('Finished in {} seconds'.format(t1-t0))

init_co2 = sum(values[0, 1] for values in data.list)
if status == pywraplp.Solver.OPTIMAL:
    print('Total CO2 reduction:', init_co2 + objective.Value())
    expenses = 0
    for i, J in enumerate(data.alternatives_per_individual):
        for j in range(J):
            if x[i, j].solution_value() > 0:
                expenses += data.list[i][j, 0]
    print('Total expenses:', expenses)
else:
    print('The problem does not have an optimal solution.')
