"""Script to prepare data before running the algorithm.

The `data/` directory must contain the predictions from the regression (script
05).

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    'data/long_data_rhone.csv',
    usecols=['obs_id', 'alt_id', 'choice_id', 'choice', 'kg_co2'],
    dtype={'alt_id': 'category'},
)
utilities = pd.read_csv('data/predictions.csv', index_col=0)

df['utility'] = np.nan
df.loc[df['alt_id'] == 'cycling', 'utility'] = utilities['cycling'].values
df.loc[df['alt_id'] == 'car', 'utility'] = utilities['car'].values
df.loc[df['alt_id'] == 'motorcycle', 'utility'] = \
    utilities['motorcycle'].values
df.loc[df['alt_id'] == 'public transit', 'utility'] = \
    utilities['public transit'].dropna().values
df.loc[df['alt_id'] == 'walking', 'utility'] = utilities['walking'].values

# Generate epsilons.
df['epsilon'] = 0
df['true_utility'] = np.nan
df['is_valid'] = False
n = len(df)
while df['is_valid'].sum() + 500 < n:
    mask = ~df['is_valid']
    m = mask.sum()
    print('Remaining: {}'.format(m))
    df.loc[mask, 'epsilon'] = np.random.gumbel(size=m)
    df.loc[mask, 'true_utility'] = (
        df.loc[mask, 'utility'] + df.loc[mask, 'epsilon']
    )
    max_utilities = df.loc[mask].groupby('obs_id')['true_utility'].max()
    real_utilities = pd.Series(df.loc[
        mask & df['choice'], ['obs_id', 'true_utility']
    ].set_index('obs_id')['true_utility'])
    valid_obs = real_utilities.index[max_utilities == real_utilities]
    df.loc[df['obs_id'].isin(valid_obs), 'is_valid'] = True

for obs in df['obs_id'].unique():
    mask = df['obs_id'] == obs
    if not df.loc[mask, 'is_valid'].all():
        r = 0
        utilities = df.loc[mask, 'utility'].values
        J = len(utilities)
        j = np.argmax(df.loc[mask, 'choice'])
        while True:
            r += 1
            epsilons = np.random.gumbel(size=J)
            true_utilities = utilities + epsilons
            if np.max(true_utilities) == utilities[j] + epsilons[j]:
                df.loc[mask, 'epsilon'] = epsilons
                df.loc[mask, 'true_utility'] = true_utilities
                df.loc[mask, 'is_valid'] = True
                m -= J
                print(obs)
                print('Number of tries: {}'.format(r))
                print('Remaining: {}'.format(m))
                break

df = df.drop(columns='is_valid')

# Plot epsilons distribution.
fig, ax = plt.subplots()
ax.hist(df['epsilon'], bins=50, density=True)
xs = np.linspace(df['epsilon'].min(), df['epsilon'].max(), 200)
ys = np.exp(-(xs + np.exp(-xs)))
ax.plot(xs, ys, label='Gumbel density')
ax.set_xlabel('Value')
ax.set_ylabel('Frequence')
ax.legend()
fig.tight_layout()
fig.savefig('output/epsilon_hist.pdf')

# Convert utility to euros.
utility_vot = 1.876496  # average value of time, from the regressions
euro_vot = 9.17  # average value of time in euros, from literature
mu = euro_vot / utility_vot
print('Value of one utility unit: {} euros'.format(mu))
df['utility'] = df['utility'] * mu
df['true_utility'] = df['true_utility'] * mu

# Compute the opposite of the expectation of the minimum incentive needed.
df['default_utility'] = np.repeat(
    df.loc[df['choice'], 'utility'].values,
    df.groupby('obs_id')['alt_id'].count()
)
df['minus_incentive'] = -(
    mu * (1 + np.exp((df['utility']-df['default_utility'])/mu))
    * np.log(1 + np.exp((df['default_utility']-df['utility'])/mu))
)
df.loc[df['choice'], 'minus_incentive'] = 0

df.to_csv('data/algorithm_input_rhone.csv', index=False)
