"""Script to prepare the regressions.

The `data/` directory must contain the cleaned census data (script 01) and the
OD matrix with travel times (script 03).

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylogit import conditional_logit

# Create directory `output/` if it does not exist.
if not os.path.isdir('output'):
    os.mkdir('output')

# Open the data.
df = pd.read_csv(
    'data/population_rhone.csv',
    usecols=[
        'AGEREVQ', 'CS1', 'DIPL', 'EMPL', 'IPONDI', 'SEXE', 'TRANS', 'home',
        'work', 'car_per_indiv',
    ],
    dtype={'home': str, 'work': str, 'CS1': 'category', 'DIPL': 'category',
           'EMPL': 'category', 'SEXE': 'category', 'TRANS': 'category'},
)
od_matrix = pd.read_csv(
    'data/od_matrix_rhone.csv',
    dtype={'home': str, 'work': str},
)

# Merge travel times with the census data.
df = df.merge(od_matrix, on=['home', 'work'], how='left')

# Clean data.
df = df.loc[df['home'].str.startswith('69')]
df = df.loc[df['work'].str.startswith('69')]
n0 = len(df)
w0 = df['IPONDI'].sum()
df = df.loc[df['tt_walking'].notna()]
df = df.loc[df['tt_cycling'].notna()]
df = df.loc[df['tt_motorcycle'].notna()]
df = df.loc[df['tt_car'].notna()]
n1 = n0 - len(df)
w1 = w0 - df['IPONDI'].sum()
print((
    'There are {} out of {} individuals with invalid travel times '
    '(representing {:.2%} of weight)'
).format(n1, n0, w1/w0))

n0 = len(df)
w0 = df['IPONDI'].sum()
n1 = len(df.loc[df['home']==df['work']])
w1 = df.loc[df['home']==df['work'], 'IPONDI'].sum()
print((
    'There are {} out of {} individuals with intra-city trip '
    '(representing {:.2%} of weight)'
).format(n1, n0, w1/w0))

df['SEXE'] = df['SEXE'] == 'Femme'

# Convert to long data.
long_df = pd.DataFrame()
nb_modes = 5
long_df['obs_id'] = np.repeat(np.arange(len(df)), nb_modes)
long_df['alt_id'] = np.tile(np.arange(nb_modes), len(df))
long_df['home'] = np.repeat(df['home'], nb_modes).values
long_df['work'] = np.repeat(df['work'], nb_modes).values
long_df['AGEREVQ'] = np.repeat(df['AGEREVQ'], nb_modes).values
long_df['CS1'] = np.repeat(df['CS1'], nb_modes).values
long_df['DIPL'] = np.repeat(df['DIPL'], nb_modes).values
long_df['EMPL'] = np.repeat(df['EMPL'], nb_modes).values
long_df['IPONDI'] = np.repeat(df['IPONDI'], nb_modes).values
long_df['SEXE'] = np.repeat(df['SEXE'], nb_modes).astype(int).values
long_df['car_per_indiv'] = np.repeat(df['car_per_indiv'], nb_modes).values
long_df['choice_id'] = np.repeat(df['TRANS'].cat.codes, nb_modes).values
long_df['choice'] = long_df['choice_id'] == long_df['alt_id']
long_df['tt'] = np.nan
long_df['kg_co2'] = np.nan
for i, alt in enumerate(df['TRANS'].cat.categories):
    alt = alt.replace(' ', '_')
    # Travel times and CO2 are doubled to account for morning-evening commute.
    long_df.loc[long_df['alt_id']==i, 'tt'] = 2 * df['tt_{}'.format(alt)].values
    if alt in ('cycling', 'walking'):
        long_df.loc[long_df['alt_id']==i, 'kg_co2'] = 0
    else:
        long_df.loc[
            long_df['alt_id']==i, 'kg_co2'
        ] = 2 * df['kg_co2_{}'.format(alt)].values
    long_df['tt_{}'.format(alt)] = 0
    long_df.loc[long_df['alt_id']==i, 'tt_{}'.format(alt)] = \
        2 * df['tt_{}'.format(alt)].values
    long_df['tt_{}_2'.format(alt)] = 0
    long_df.loc[long_df['alt_id']==i, 'tt_{}_2'.format(alt)] = \
        2 * df['tt_{}'.format(alt)].values ** 2
    long_df['log_tt_{}'.format(alt)] = 0
    long_df.loc[long_df['alt_id']==i, 'log_tt_{}'.format(alt)] = \
        2 * np.log( df['tt_{}'.format(alt)].values + 1 )
long_df['tt_x_AGEREVQ'] = long_df['tt'] * long_df['AGEREVQ']
long_df['tt_x_SEXE'] = long_df['tt'] * long_df['SEXE']
long_df['tt_x_artisan'] = long_df['tt'] * (long_df['CS1'] == 'Artisan')
long_df['tt_x_bluecollar'] = long_df['tt'] * (long_df['CS1'] == 'Blue-collar')
long_df['tt_x_employee'] = long_df['tt'] * (long_df['CS1'] == 'Employee')
long_df['tt_x_executive'] = long_df['tt'] * (long_df['CS1'] == 'Executive')
long_df['tt_x_farmer'] = long_df['tt'] * (long_df['CS1'] == 'Farmer')
long_df['tt_x_intermediate'] = long_df['tt'] * (long_df['CS1'] == 'Intermediate')
n0 = long_df['obs_id'].nunique()
w0 = long_df.groupby('obs_id')['IPONDI'].mean().sum()
long_df = long_df.dropna()
valid_obs = np.flatnonzero(long_df.groupby('obs_id')['alt_id'].count() == 4)
n1 = len(valid_obs)
w1 = long_df.loc[long_df['obs_id'].isin(valid_obs)].groupby('obs_id')['IPONDI'].mean().sum()
print((
    'There are {} out of {} individuals with no access to public transit '
    '(representing {:.2%} of weight)'
).format(n1, n0, w1/w0))

# Drop individuals taking public transit if no itinerary was found.
valid_obs = long_df.groupby('obs_id')['choice'].sum() == 1
long_df = long_df.loc[long_df['obs_id'].isin(np.flatnonzero(valid_obs))]
# Rename categories.
long_df['alt_id'] = long_df['alt_id'].astype('category').cat.rename_categories(
    ['car', 'cycling', 'motorcycle', 'public transit', 'walking']
)
# Set number of cars to zero for non-car alternatives.
long_df.loc[long_df['alt_id']!='car', 'car_per_indiv'] = 0
# Compute positive car number indicator.
long_df['has_car'] = long_df['car_per_indiv'] > 0
# Drop individuals with travel time larger than 90 minutes.
n0 = long_df['obs_id'].nunique()
w0 = long_df.groupby('obs_id')['IPONDI'].mean().sum()
valid_obs = long_df.loc[(long_df['choice'])&(long_df['tt']<=2*1.5), 'obs_id'].values
long_df = long_df.loc[long_df['obs_id'].isin(valid_obs)]
n1 = n0 - long_df['obs_id'].nunique()
w1 = w0 - long_df.groupby('obs_id')['IPONDI'].mean().sum()
print((
    'There are {} out of {} individuals with too large travel times '
    '(representing {:.2%} of weight)'
).format(n1, n0, w1/w0))

fig, ax = plt.subplots()
xs = np.arange(1.25, 91, 2.5)
bins = np.append(xs-1.25, 92.5)
mode = 'walking'
valid_df = long_df.loc[long_df['choice']]
for mode in ('car', 'public transit', 'walking', 'cycling', 'motorcycle'):
    valids = valid_df.loc[valid_df['alt_id']==mode, 'tt'].values
    totals = long_df.loc[long_df['alt_id']==mode, 'tt'].values
    valid_means, _ = np.histogram(valids*60/2, bins)
    total_means, _ = np.histogram(totals*60/2, bins)
    ax.plot(xs, valid_means/total_means, label=mode)
del valid_df
ax.set_xlabel('Travel time (min.)')
ax.set_ylabel('Share')
ax.legend()
fig.tight_layout()
fig.savefig('output/tt_share.pdf')

long_df.to_csv('data/long_data_rhone.csv', index=False)
