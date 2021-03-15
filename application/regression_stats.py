"""Compute some statistics from the regression results.

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

res = pd.read_csv('output/regression_results.csv')
res = pd.Series(res.set_index('Unnamed: 0')['x'])

fig, ax = plt.subplots()
xs = np.linspace(0, 1.5, 200)
ys_car = (
    res['car_per_indiv'] + res['has_carTRUE'] + res['tt_car'] * xs
    + res['tt_x_AGEREVQ'] * xs * 40
)
if 'tt_car_2' in res:
    ys_car += res['tt_car_2'] * xs**2
if 'log_tt_car' in res:
    ys_car += res['log_tt_car'] * np.log(xs)
ys_public = (
    res['(Intercept):public transit'] + res['AGEREVQ:public transit'] * 40
    + res['tt_public_transit'] * xs + res['tt_x_AGEREVQ'] * xs * 40
)
if 'tt_public_transit_2' in res:
    ys_public += res['tt_public_transit_2'] * xs**2
if 'log_tt_public_transit' in res:
    ys_public += res['log_tt_public_transit'] * np.log(xs)
ys_walking = (
    res['(Intercept):walking'] + res['AGEREVQ:walking'] * 40 +
    res['tt_walking'] * xs + res['tt_x_AGEREVQ'] * xs * 40
)
if 'tt_walking_2' in res:
    ys_walking += res['tt_walking_2'] * xs**2
if 'log_tt_walking' in res:
    ys_walking += res['log_tt_walking'] * np.log(xs)
ys_cycling = (
    res['(Intercept):cycling'] + res['AGEREVQ:cycling'] * 40 +
    res['tt_cycling'] * xs + res['tt_x_AGEREVQ'] * xs * 40
)
if 'tt_cycling_2' in res:
    ys_cycling += res['tt_cycling_2'] * xs**2
if 'log_tt_cycling' in res:
    ys_cycling += res['log_tt_cycling'] * np.log(xs)
ys_motorcycle = (
    res['(Intercept):motorcycle'] + res['AGEREVQ:motorcycle'] * 40 +
    res['tt_motorcycle'] * xs + res['tt_x_AGEREVQ'] * xs * 40
)
if 'tt_motorcycle_2' in res:
    ys_motorcycle += res['tt_motorcycle_2'] * xs**2
if 'log_tt_motorcycle' in res:
    ys_motorcycle += res['log_tt_motorcycle'] * np.log(xs)
ax.plot(xs, ys_car, label='car', alpha=.7)
ax.plot(xs, ys_public, label='public transit', alpha=.7)
ax.plot(xs, ys_walking, label='walking', alpha=.7)
ax.plot(xs, ys_cycling, label='cycling', alpha=.7)
ax.plot(xs, ys_motorcycle, label='motorcycle', alpha=.7)
ax.legend()
ax.set_xlim(0, 1.5)
ax.set_xlabel('Travel time (hour)')
ax.set_ylabel('Utility')
fig.tight_layout()
fig.savefig('output/tt_utility.pdf')
