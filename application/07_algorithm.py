"""Script to run the MCKP algorithm.

The `data/` directory must contain the algorithm input (script 04).

Author: Lucas Javaudin
E-mail: lucas.javaudin@ens-paris-saclay.fr
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
print('Number of individuals: {}'.format(data.individuals))
print('Number of alternatives: {}'.format(
    np.sum(data.alternatives_per_individual)))

print('Running algorithm')
budget = 3000
# budget = np.inf
results = data.run_lite_algorithm(budget=budget, verbose=False)
print('Computing results')
results.compute_results()
results.output_characteristics('output/results_characteristics.txt')
results.output_results('output/results.txt')
results.plot_choice_share('output/choice_share.pdf', show_title=False)

# Maximum social welfare curve.
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
ax.step(results.expenses_history, results.total_energy_gains_history/1000,
        where='post', color=COLOR_1,
        label=r'Maximum social welfare curve $\mathcal{C}_Q$')
ax.set_xlabel('Daily budget $Y^{[k]}$ (euros)')
ax.set_ylabel('CO2 reduction (tons)')
ax.set_xlim(left=0, right=results.expenses)
ax.set_ylim(bottom=0)
ax.legend(loc='lower right')
ax.grid()
fig.tight_layout()
fig.savefig('output/max_soc_welfare_curve.pdf')

# Inverse Efficiency curve.
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
# Marginal euro / ton of co2
ys = 1000 / results.efficiencies_history
ys = np.append(0, ys)
ax.plot(results.expenses_history, ys, color=COLOR_1, linestyle='dashed',
        label=r'Inverse incremental efficiency $1/\tilde{e}^{[k]}$')
# Mean euro / ton of co2
ys = (
    1000 * results.expenses_history / results.total_energy_gains_history)
ys[0] = 0
ax.plot(results.expenses_history, ys, color=COLOR_2,
        label=r'Inverse overall efficiency $1/e^{[k]}$')
ax.set_xlabel('Daily budget $Y^{[k]}$ (euros)')
ax.set_ylabel('Inverse efficiency (euros per ton of CO2)')
ax.set_xlim(left=0, right=results.expenses)
ax.set_ylim(bottom=0)
ax.grid()
ax.legend()
fig.tight_layout()
fig.savefig('output/inverse_efficiency.pdf')
if np.max(ys) >= 100:
    b = results.expenses_history[np.argmax(ys>=100)]
    print('Budget corresponding to 100 euros per ton of co2: {}'.format(b))

jumps_history = list()
id_to_case = input_df['obs_id'].unique()
for i, prev_j, next_j in results.jumps_history:
    case = id_to_case[i]
    prev_alt = data.alternative_labels[i][prev_j]
    next_alt = data.alternative_labels[i][next_j]
    jumps_history.append([case, prev_alt, next_alt])

jumps = dict()
for i in range(results.iteration):
    indiv, prev_j, next_j = jumps_history[i]
    incentive = results.incentives_history[i]
    energy_gain = results.energy_gains_history[i]
    if indiv in jumps:
        jumps[indiv][1] = next_j
        jumps[indiv][2] += incentive
        jumps[indiv][3] += energy_gain
    else:
        jumps[indiv] = [prev_j, next_j, incentive, energy_gain]
tmp_df = df.copy()
for key, item in jumps.items():
    tmp_df.loc[(key, item[0]), 'choice'] = False
    tmp_df.loc[(key, item[1]), 'choice'] = True
last_state = tmp_df.loc[tmp_df['choice']].index
del tmp_df
jumps = pd.DataFrame(jumps).T
jumps = jumps.rename(columns={2: 'incentive', 3: 'energy_gain'})
jumps_str = pd.Series([' -> '.join(row[:2]) for key, row in jumps.iterrows()])
jumps_str = jumps_str.astype('category')
jump_types, counts = np.unique(jumps_str, return_counts=True)
for jump_type, count in zip(jump_types, counts):
    print('Jump {}: {} individuals ({:.4%})'.format(jump_type, count,
                                                      count/data.individuals))
print('Number of incentives given: {}'.format(len(jumps)))
print('({:.4%} of all individuals)'.format(len(jumps)/data.individuals))
print('Average amount of incentives: {}'.format(jumps['incentive'].mean()))
print('Average energy gain: {}'.format(jumps['energy_gain'].mean()))

# Jumps scatter.
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

fig = plt.figure(figsize=set_size(ratio=1, fraction=.8))
ax_scatter = fig.add_axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = fig.add_axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = fig.add_axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)

ax_scatter.scatter(jumps['incentive'], jumps['energy_gain'], alpha=.5, s=1)
slope = results.last_efficiency
xs = [0, max(jumps['incentive'].max(), jumps['energy_gain'].max()/slope)]
ys = [0, max(jumps['energy_gain'].max(), jumps['incentive'].max()*slope)]
ax_scatter.plot(xs, ys, color='black')
ax_scatter.set_xlim(0, jumps['incentive'].max())
ax_scatter.set_ylim(0, jumps['energy_gain'].max())

ax_histx.hist(jumps['incentive'], bins=30)
ax_histy.hist(jumps['energy_gain'], bins=30, orientation='horizontal')
ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())

ax_scatter.set_xlabel('Incentive amount (euro)')
ax_scatter.set_ylabel('CO2 reduction (kg.)')
fig.savefig('output/jumps_scatter.pdf')

# Switch heatmap.
w = np.zeros([5, 5], dtype=np.float64)
alts = ['car', 'public transit', 'walking', 'cycling', 'motorcycle']
for i, from_alt in enumerate(alts):
    w[i, i] = (input_df.loc[input_df['choice'], 'alt_id'] == from_alt).sum()
    for j, to_alt in enumerate(alts):
        if i != j:
            n = len(jumps.loc[(jumps[0]==from_alt)&(jumps[1]==to_alt)])
            w[i, i] -= n
            w[i, j] = n
w /= data.individuals
w = np.append(w, np.expand_dims(w.sum(axis=0), 0), axis=0)
w = np.append(w, np.expand_dims(w.sum(axis=1), 1), axis=1)
alts.append('total')
fig, ax = plt.subplots(figsize=set_size(ratio=1, fraction=.75))
ax.imshow(w+1e-3, norm=colors.LogNorm(vmin=w.min()+1e-3, vmax=w.max()+1e-3))
ax.set_xlabel('Mode choice after the policy')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Mode choice before the policy')
ax.set_xticks(np.arange(len(alts)))
ax.set_yticks(np.arange(len(alts)))
ax.set_xticklabels(alts)
ax.set_yticklabels(alts)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.xticks(rotation=90)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([4.5], minor=True)
ax.set_yticks([4.5], minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(top=False, left=False)
ax.tick_params(which="minor", bottom=False, left=False)
for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        if w[i, j] > .01:
            c = 'black'
        else:
            c = 'white'
        text = ax.text(j, i, r'{:g}\%'.format(round(100*w[i, j], 3)),
                       ha='center', va='center', color=c)
fig.tight_layout()
fig.savefig('output/switches_heatmap.pdf')

# Travel times change curve.
jumps['tt_0'] = np.nan
jumps['tt_1'] = np.nan
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
xs = results.expenses_history
ys = [df.loc[df['choice'], 'tt'].sum()]
for i, prev_j, next_j in jumps_history:
    prev_tt = df.loc[(i, prev_j), 'tt']
    next_tt = df.loc[(i, next_j), 'tt']
    jumps.loc[i, 'tt_0'] = prev_tt
    jumps.loc[i, 'tt_1'] = next_tt
    ys.append(ys[-1]-prev_tt+next_tt)
ys = np.array(ys) / data.individuals
ys = ys * 60 # Compute average travel time in minutes.
ys = ys / 2 # Compute average travel time for only one trip.
ax.plot(xs, ys, color=COLOR_1)
ax.set_xlabel('Expenses (euros)')
ax.set_ylabel('Average travel times (min.)')
ax.set_xlim(left=0)
ax.grid()
fig.tight_layout()
fig.savefig('output/tt_evolution.pdf')

# Travel times change scatter plot.
cmap = plt.get_cmap('tab10')
alts = ['car', 'public transit', 'walking', 'cycling', 'motorcycle']
alt_to_color = dict()
for i, alt in enumerate(alts):
    alt_to_color[alt] = cmap(i)
for i, from_alt in enumerate(['car', 'public transit', 'motorcycle']):
    fig, ax = plt.subplots(figsize=set_size(fraction=.8))
    for j, to_alt in enumerate(alts):
        if len(jumps.loc[(jumps[0]==from_alt)&(jumps[1]==to_alt)]):
            ax.scatter(
                jumps.loc[(jumps[0]==from_alt)&(jumps[1]==to_alt), 'tt_0']*60/2,
                jumps.loc[(jumps[0]==from_alt)&(jumps[1]==to_alt), 'tt_1']*60/2,
                color=alt_to_color[to_alt], s=5, marker='.', alpha=.9,
                label=to_alt,
            )
    ax.plot([0, 90], [0, 90], color='black')
    ax.legend(title='New mode')
    ax.set_xlabel('Previous travel time (min.)')
    ax.set_ylabel('New travel time (min.)')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    fig.tight_layout()
    fig.savefig('output/tt_scatter_{}.pdf'.format(from_alt.replace(' ', '_')))

# Travel times change histogram.
from_alt = 'car'
valid_jumps = jumps.loc[jumps[0]==from_alt].copy()
valid_jumps['tt_diff'] = (valid_jumps['tt_1'] - valid_jumps['tt_0']) * 60 / 2
x = np.arange(valid_jumps['tt_diff'].min(), valid_jumps['tt_diff'].max()+1, 5)
bins = np.append(x-2.5, x[-1]+2.5)
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
n = valid_jumps[1].nunique()
width = 4
for i, to_alt in enumerate(valid_jumps[1].unique()):
    means, _ = np.histogram(
        valid_jumps.loc[valid_jumps[1]==to_alt, 'tt_diff'], bins)
    ax.bar(x-width/2+i*width/n, means, width/n, align='edge', label=to_alt,
           color=alt_to_color[to_alt])
# ax.set_xticks(np.rint(x))
# ax.set_xticklabels(np.rint(x))
ax.set_xlabel('Change in travel time (min.)')
ax.set_ylabel('Count')
ax.legend()
fig.tight_layout()
fig.savefig('output/tt_diff_hist.pdf')

# Travel times histograms.
x = np.arange(0, 91, 5)
bins = np.append(x-2.5, 92.5)
cycling_values = df.loc(axis=0)[:, 'cycling'].loc[df['choice'], 'tt'].values
car_values = df.loc(axis=0)[:, 'car'].loc[df['choice'], 'tt'].values
motorcycle_values = df.loc(axis=0)[:, 'motorcycle'].loc[df['choice'], 'tt'].values
public_transit_values = df.loc(axis=0)[:, 'public transit'].loc[df['choice'], 'tt'].values
walking_values = df.loc(axis=0)[:, 'walking'].loc[df['choice'], 'tt'].values
cycling_means, _ = np.histogram(cycling_values*60/2, bins)
car_means, _ = np.histogram(car_values*60/2, bins)
motorcycle_means, _ = np.histogram(motorcycle_values*60/2, bins)
public_transit_means, _ = np.histogram(public_transit_values*60/2, bins)
walking_means, _ = np.histogram(walking_values*60/2, bins)
width = .8
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
ax.bar(x - 2*width, car_means, width, align='edge', label='car')
ax.bar(x - 1.2*width, public_transit_means, width, align='edge', label='public transit')
ax.bar(x - .4*width, walking_means, width, align='edge', label='walking')
ax.bar(x + .4*width, cycling_means, width, align='edge', label='cycling')
ax.bar(x + 1.2*width, motorcycle_means, width, align='edge', label='motorcycle')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_xlabel('Travel time (min.)')
ax.set_ylabel('Count')
ax.legend()
fig.tight_layout()
fig.savefig('output/tt_hist_init.pdf')

x = np.arange(0, 91, 5)
bins = np.append(x-2.5, 92.5)
cycling_values = df.loc[last_state].loc(axis=0)[:, 'cycling']['tt'].values
car_values = df.loc[last_state].loc(axis=0)[:, 'car']['tt'].values
motorcycle_values = df.loc[last_state].loc(axis=0)[:, 'motorcycle']['tt'].values
public_transit_values = df.loc[last_state].loc(axis=0)[:, 'public transit']['tt'].values
walking_values = df.loc[last_state].loc(axis=0)[:, 'walking']['tt'].values
cycling_means, _ = np.histogram(cycling_values*60/2, bins)
car_means, _ = np.histogram(car_values*60/2, bins)
motorcycle_means, _ = np.histogram(motorcycle_values*60/2, bins)
public_transit_means, _ = np.histogram(public_transit_values*60/2, bins)
walking_means, _ = np.histogram(walking_values*60/2, bins)
width = .8
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
ax.bar(x - 2, car_means, width, align='edge', label='car')
ax.bar(x - 1.2, public_transit_means, width, align='edge', label='public transit')
ax.bar(x - .4, walking_means, width, align='edge', label='walking')
ax.bar(x + .4, cycling_means, width, align='edge', label='cycling')
ax.bar(x + 1.2, motorcycle_means, width, align='edge', label='motorcycle')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_xlabel('Travel time (min.)')
ax.set_ylabel('Count')
ax.legend()
fig.tight_layout()
fig.savefig('output/tt_hist_last.pdf')

# Distance histograms.
x = np.arange(0, df['geodesic_distance'].max()+1, 5000)
bins = np.append(x-2500, x.max()+2500)
cycling_values = df.loc(axis=0)[:, 'cycling'].loc[df['choice'], 'geodesic_distance'].values
car_values = df.loc(axis=0)[:, 'car'].loc[df['choice'], 'geodesic_distance'].values
motorcycle_values = df.loc(axis=0)[:, 'motorcycle'].loc[df['choice'], 'geodesic_distance'].values
public_transit_values = \
    df.loc(axis=0)[:, 'public transit'].loc[df['choice'], 'geodesic_distance'].values
walking_values = df.loc(axis=0)[:, 'walking'].loc[df['choice'], 'geodesic_distance'].values
cycling_means, _ = np.histogram(cycling_values, bins)
car_means, _ = np.histogram(car_values, bins)
motorcycle_means, _ = np.histogram(motorcycle_values, bins)
public_transit_means, _ = np.histogram(public_transit_values, bins)
walking_means, _ = np.histogram(walking_values, bins)
width = .8
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
x /= 1000 # xscale is in kilometers.
ax.bar(x - 2, car_means, width, align='edge', label='car')
ax.bar(x - 1.2, public_transit_means, width, align='edge', label='public transit')
ax.bar(x - .4, walking_means, width, align='edge', label='walking')
ax.bar(x + .4, cycling_means, width, align='edge', label='cycling')
ax.bar(x + 1.2, motorcycle_means, width, align='edge', label='motorcycle')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.set_xlabel('Distance (km.)')
ax.set_ylabel('Count')
ax.legend()
fig.tight_layout()
fig.savefig('output/dist_hist.pdf')

fig, ax = plt.subplots(figsize=set_size(fraction=.8))
init_df = df.loc[df['choice']]
init_df = init_df.merge(
    input_df, left_index=True, right_on=['obs_id', 'alt_id'], how='inner')
init_mean_far = init_df.loc[init_df['CS1']=='Farmer', 'true_utility'].mean()
init_mean_art = init_df.loc[init_df['CS1']=='Artisan', 'true_utility'].mean()
init_mean_exe = init_df.loc[init_df['CS1']=='Executive', 'true_utility'].mean()
init_mean_int = init_df.loc[init_df['CS1']=='Intermediate', 'true_utility'].mean()
init_mean_emp = init_df.loc[init_df['CS1']=='Employee', 'true_utility'].mean()
init_mean_blu = init_df.loc[init_df['CS1']=='Blue-collar', 'true_utility'].mean()
del init_df
opt_df = df.loc[last_state]
opt_df = opt_df.merge(
    input_df, left_index=True, right_on=['obs_id', 'alt_id'], how='inner')
opt_mean_far = opt_df.loc[opt_df['CS1']=='Farmer', 'true_utility'].mean()
opt_mean_art = opt_df.loc[opt_df['CS1']=='Artisan', 'true_utility'].mean()
opt_mean_exe = opt_df.loc[opt_df['CS1']=='Executive', 'true_utility'].mean()
opt_mean_int = opt_df.loc[opt_df['CS1']=='Intermediate', 'true_utility'].mean()
opt_mean_emp = opt_df.loc[opt_df['CS1']=='Employee', 'true_utility'].mean()
opt_mean_blu = opt_df.loc[opt_df['CS1']=='Blue-collar', 'true_utility'].mean()
del opt_df
ax.bar(
    np.arange(-.5, 15, 3),
    [init_mean_far, init_mean_art, init_mean_exe, init_mean_int, init_mean_emp,
     init_mean_blu],
    width=1,
    label='Before',
)
ax.bar(
    np.arange(.5, 16, 3),
    [opt_mean_far, opt_mean_art, opt_mean_exe, opt_mean_int, opt_mean_emp,
     opt_mean_blu],
    width=1,
    label='After',
)
ax.set_xticks([0, 3, 6, 9, 12, 15])
ax.set_xticklabels(['Farmer', 'Artisan', 'Executive', 'Intermediate',
                    'Employee', 'Blue-collar'])
ax.set_xlabel('Profession')
ax.set_ylabel('Average utility')
ax.legend()
fig.tight_layout()
fig.savefig('output/utility_hist.pdf')

# Share of switchers.
car_df = df.loc(axis=0)[:, 'car'].loc[df['choice']]
counts = car_df.loc[jumps.index].groupby(['home', 'work'])['AGEREVQ'].count()
raw_counts = car_df.groupby(['home', 'work'])['AGEREVQ'].count()
shares = (counts / raw_counts).fillna(0)
fig, ax = plt.subplots(figsize=set_size(fraction=.8))
ax.hist(shares[shares>0].values, bins=np.linspace(0, 1, 41),
        weights=raw_counts[shares>0].values)
ax.set_xlabel('Share of individuals who switched in the OD pair')
ax.set_ylabel('Number of individuals')
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig('output/switches_shares.pdf')

if False:
    # Tax.
    utility_diffs = list()
    co2_gains = list()
    for key, row in input_df.loc[
            (input_df['alt_id']=='car')&(input_df['choice'])].iterrows():
        tmp_df = input_df.loc[input_df['obs_id']==row['obs_id']]
        utilities = tmp_df['true_utility'].values
        co2 = tmp_df['kg_co2'].values
        second_alt_id = np.argsort(utilities)[1]
        utility_diffs.append(row['true_utility'] - utilities[second_alt_id])
        co2_gains.append(row['kg_co2'] - co2[second_alt_id])
    utility_diffs = np.array(utility_diffs)
    co2_gains = np.array(co2_gains)
    idx = np.argsort(utility_diffs)
    utility_diffs = utility_diffs[idx]
    co2_gains = co2_gains[idx]
    expenses = utility_diffs * (1+np.arange(len(utility_diffs)))
    co2_gains = np.cumsum(co2_gains)
    # Plot.
    fig, ax = plt.subplots()
    ax.plot(expenses, co2_gains)
    ax.set_xlabel('Expenses (euros)')
    ax.set_ylabel('Reduction in CO2 emissions (kg.)')
    fig.tight_layout()
    fig.savefig('output/flat_subsidy.pdf')
    # Summary.
    i = np.argmax(expenses>budget) - 1
    print(
        (
            'With a flat subsidy of {} euros to stop using car, {} individuals '
            'would accept the subsidy for total expenses of {} euros and a '
            'reduction in CO2 emissions of {} kilograms ({:.4%} of the incentive '
            'policy).'
        ).format(utility_diffs[i], i+1, expenses[i], co2_gains[i],
                 co2_gains[i]/results.total_energy_gains)
    )
    utility_inc = expenses[i] - np.sum(utility_diffs[:i+1])
    print('Individual utility would increase by: {}'.format(utility_inc))
