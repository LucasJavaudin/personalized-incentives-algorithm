"""Script to compute travel times for all modes of transportation, for all
individuals.

The `data/` directory must contain a file with the OD matrix (script 01) and a
file with the town hall coordinates (script 02).

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import os
import json
from zipfile import ZipFile

import requests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from geopy.distance import distance
import statsmodels.api as sm

HERE_API_KEY = 'YOUR_HERE_API_KEY'

##################
#  Import Data.  #
##################

# Open the OD matrix.
df = pd.read_csv(
    'data/od_matrix_rhone.csv',
    dtype={'home': str, 'work': str},
)
# Open the town hall geojson.
townhall_df = pd.read_csv(
    'data/admin_centre.csv',
    dtype={'insee': str},
)

# Check that all home and work locations have an administrative center.
invalid_insee = set(
    df.loc[~df['home'].isin(townhall_df['insee']), 'home'].unique()
)
invalid_insee = invalid_insee.union(
    df.loc[~df['work'].isin(townhall_df['insee']), 'work'].unique())
if len(invalid_insee):
    print('Warning. The following INSEE codes have not been found, using city-'
          'center instead: {}'.format(invalid_insee))
    # Read commune geometries.
    geo_df = gpd.read_file('data/communes-20190101.shp')
    geo_df = geo_df.loc[geo_df['insee'].isin(invalid_insee)]
    geo_df = geo_df.set_crs(epsg=4326).to_crs(epsg=2154)
    geo_df['lon'] = geo_df.geometry.centroid.to_crs(epsg=4326).x
    geo_df['lat'] = geo_df.geometry.centroid.to_crs(epsg=4326).y
    geo_df = pd.DataFrame(geo_df[['insee', 'lat', 'lon']])
    townhall_df = townhall_df.append(geo_df)
    # Read district geometries.
    geo_df = gpd.read_file('data/arrondissements-municipaux-20160128.shp')
    geo_df = geo_df.loc[geo_df['insee'].isin(invalid_insee)]
    geo_df = geo_df.set_crs(epsg=4326).to_crs(epsg=2154)
    geo_df['lon'] = geo_df.geometry.centroid.to_crs(epsg=4326).x
    geo_df['lat'] = geo_df.geometry.centroid.to_crs(epsg=4326).y
    geo_df = pd.DataFrame(geo_df[['insee', 'lat', 'lon']])
    townhall_df = townhall_df.append(geo_df)

# Merge the DataFrames.
df = df.merge(townhall_df[['insee', 'lat', 'lon']], how='left',
              left_on='home', right_on='insee')
df = df.merge(townhall_df[['insee', 'lat', 'lon']], how='left',
              left_on='work', right_on='insee',
              suffixes=('_home', '_work'))
df = df.drop(columns=['insee_home', 'insee_work'])

if np.any(df.isna().sum()):
    raise Exception('Some INSEE codes have not been found.')

# Remove OD pairs which are too far away.
df['geodesic_distance'] = [
    distance((row['lat_home'], row['lon_home']),
             (row['lat_work'], row['lon_work'])).meters
    for key, row in df.iterrows()
]
to_be_removed = df['geodesic_distance'] > 150e3 # 150 kilometers.
removed_pairs = np.sum(to_be_removed)
removed_indivs = df.loc[to_be_removed, 'size'].sum()
removed_weight = df.loc[to_be_removed, 'weight'].sum()
df = df.loc[~to_be_removed]
msg = ('Removed {} O-D pairs with a distance greater than 150 km '
       '(representing {} individuals and {} weight).')
print(msg.format(removed_pairs, removed_indivs, removed_weight))


##########################
#  Walking travel times  #
##########################

df['tt_walking'] = np.nan
df['dist_walking'] = np.nan

for key, row in df.iterrows():
    if row['home'] != row['work']:
        x0 = row['lat_home']
        y0 = row['lon_home']
        x1 = row['lat_work']
        y1 = row['lon_work']
        url = 'http://localhost:8989/route'
        params = {
            'point': [f'{x0},{y0}', f'{x1},{y1}'],
            'instructions': 'false',
            'vehicle': 'foot',
        }
        r = requests.get(url, params=params)
        data = json.loads(r.content)
        try:
            path = data['paths'][0]
            df.loc[key, 'tt_walking'] = path['time']
            df.loc[key, 'dist_walking'] = path['distance']
        except KeyError:
            print(data)

# Convert time from ms to hours.
df['tt_walking'] = df['tt_walking'] / (3600 * 1e3)


##########################
#  Bicycle travel times  #
##########################

df['tt_cycling'] = np.nan
df['dist_cycling'] = np.nan

for key, row in df.iterrows():
    if row['home'] != row['work']:
        x0 = row['lat_home']
        y0 = row['lon_home']
        x1 = row['lat_work']
        y1 = row['lon_work']
        url = 'http://localhost:8989/route'
        params = {
            'point': [f'{x0},{y0}', f'{x1},{y1}'],
            'instructions': 'false',
            'vehicle': 'bike',
        }
        r = requests.get(url, params=params)
        data = json.loads(r.content)
        try:
            path = data['paths'][0]
            df.loc[key, 'tt_cycling'] = path['time']
            df.loc[key, 'dist_cycling'] = path['distance']
        except KeyError:
            print(data)

# Convert time from ms to hours.
df['tt_cycling'] = df['tt_cycling'] / (3600 * 1e3)


#############################
#  Motorcycle travel times  #
#############################

df['tt_motorcycle'] = np.nan
df['dist_motorcycle'] = np.nan

for key, row in df.iterrows():
    if row['home'] != row['work']:
        x0 = row['lat_home']
        y0 = row['lon_home']
        x1 = row['lat_work']
        y1 = row['lon_work']
        url = 'http://localhost:8989/route'
        params = {
            'point': [f'{x0},{y0}', f'{x1},{y1}'],
            'instructions': 'false',
            'vehicle': 'motorcycle',
        }
        r = requests.get(url, params=params)
        data = json.loads(r.content)
        try:
            path = data['paths'][0]
            df.loc[key, 'tt_motorcycle'] = path['time']
            df.loc[key, 'dist_motorcycle'] = path['distance']
        except KeyError:
            print(data)

# Convert time from ms to hours.
df['tt_motorcycle'] = df['tt_motorcycle'] / (3600 * 1e3)


######################
#  Car travel times  #
######################

df['tt_car'] = np.nan
df['dist_car'] = np.nan

for key, row in df.iterrows():
    if row['home'] != row['work']:
        x0 = row['lat_home']
        y0 = row['lon_home']
        x1 = row['lat_work']
        y1 = row['lon_work']
        url = 'https://route.ls.hereapi.com/routing/7.2/calculateroute.json'
        params = {
            'apiKey': HERE_API_KEY,
            'waypoint0': f'{x0},{y0}',
            'waypoint1': f'{x1},{y1}',
            'mode': 'fastest;car;traffic:enabled',
            'departure': '2020-09-15T08:00:00', # Tuesday 08:00 AM.
            'alternatives': 0,
        }
        r = requests.get(url, params=params)
        data = json.loads(r.content)
        try:
            summary = data['response']['route'][0]['summary']
            df.loc[key, 'tt_car'] = summary['trafficTime']
            df.loc[key, 'dist_car'] = summary['distance']
        except KeyError:
            print(data)
            break

# Convert time from seconds to hours.
df['tt_car'] = df['tt_car'] / 3600


#################################
#  Public Transit travel times  #
#################################

df['tt_public_transit'] = np.nan
dist_cols = list()

for key, row in df.iterrows():
    print(key)
    if row['home'] != row['work']:
        x0 = row['lat_home']
        y0 = row['lon_home']
        x1 = row['lat_work']
        y1 = row['lon_work']
        url = 'https://route.ls.hereapi.com/routing/7.2/calculateroute.json'
        params = {
            'apiKey': HERE_API_KEY,
            'waypoint0': f'{x0},{y0}',
            'waypoint1': f'{x1},{y1}',
            'mode': 'fastest;publicTransportTimeTable',
            'departure': '2020-09-15T08:00:00', # Tuesday 08:00 AM.
            'alternatives': 0,
            'maneuverAttributes': ['length', 'publicTransportLine'],
        }
        r = requests.get(url, params=params)
        if r.status_code == 400:
            # No route found.
            continue
        data = json.loads(r.content)
        route = data['response']['route'][0]
        df.loc[key, 'tt_public_transit'] = route['summary']['travelTime']
        dists = dict()
        for maneuver in route['leg'][0]['maneuver']:
            if 'line' in maneuver:
                line = maneuver['line']
                if line in dists:
                    dists[maneuver['line']] += maneuver['length']
                else:
                    dists[maneuver['line']] = maneuver['length']
        for line in route['publicTransportLine']:
            if line['id'] in dists:
                col = 'dist_{}'.format(line['type'])
                if not col in dist_cols:
                    df[col] = 0
                    dist_cols.append(col)
                df.loc[key, col] += dists[line['id']]

# Convert time from seconds to hours.
df['tt_public_transit'] = df['tt_public_transit'] / 3600

#####################
#  Intracity trips  #
#####################

# Variable surf_ha is the area of the city (in hectare). We convert it to
# square-meters.
df['surf_ha'] *= 1e4
# Compute the city radius (assuming they are circular).
df['radius'] = np.sqrt(df['surf_ha'] / np.pi) # area = pi * radius^2
x_max = df.loc[df['home']==df['work'], 'radius'].max()

tt_params = dict()
dist_params = dict()
modes = ['walking', 'cycling', 'motorcycle', 'car', 'public_transit']
# df['geodesic_distance_2'] = df['geodesic_distance'] ** 2
for mode in modes:
    df2 = df.loc[df['home']!=df['work']].dropna()
    df2 = df2.loc[df['geodesic_distance']<=x_max]
    fit = sm.WLS(
        endog=df2['tt_{}'.format(mode)],
        exog=df2['geodesic_distance'],
        # exog=df2[['geodesic_distance', 'geodesic_distance_2']],
    ).fit()
    tt_params[mode] = fit.params
    if mode != 'public_transit':
        fit = sm.WLS(
            endog=df2['dist_{}'.format(mode)],
            exog=df2['geodesic_distance'],
            # exog=df2[['geodesic_distance', 'geodesic_distance_2']],
        ).fit()
        dist_params[mode] = fit.params
    else:
        for col in dist_cols:
            fit = sm.WLS(
                endog=df2[col],
                exog=df2['geodesic_distance'],
                # exog=df2[['geodesic_distance', 'geodesic_distance_2']],
            ).fit()
            dist_params[col.replace('dist_', '')] = fit.params
    print('Mode: {}'.format(mode))
    geo_speed = 1 / (1000*tt_params[mode]['geodesic_distance'])
    print('Average speed (per geodesic distance): {}'.format(geo_speed))
    if mode != 'public_transit':
        speed = geo_speed * 1000 * dist_params[mode]['geodesic_distance']
        print('Average speed (per distance): {}'.format(speed))
    # Plot a figure.
    fig, ax = plt.subplots()
    ax.scatter(df2['geodesic_distance']/1e3, df2['tt_{}'.format(mode)],
               alpha=.3, s=1)
    xs = np.linspace(0, x_max, 200)
    ys = (
        tt_params[mode]['geodesic_distance'] * xs
        # + tt_params[mode]['geodesic_distance_2'] * xs**2
    )
    ax.plot(xs/1e3, ys, color='red')
    ax.set_xlabel('Geodesic distance (kilometers)')
    ax.set_ylabel('Travel time (hour)')
    fig.tight_layout()
    fig.savefig('output/dist_tt_{}.png'.format(mode))
# Average speed walking: 3.9 (5.0)
# Average speed cycling: 12.2 (16.8)
# Average speed motorcycle: 30.1 (46.6)
# Average speed car: 16.8 (25.7)
# Average speed public transit: 5.2

# Travel times and distances for intra-city trips.
for key, row in df.iterrows():
    if row['home'] == row['work']:
        for mode in modes:
            # Assume that geodesic distance of the trips is half the radius.
            tt = row['radius'] * tt_params[mode]['geodesic_distance'] / 2
            df.loc[key, 'tt_{}'.format(mode)] = tt
            if mode != 'public_transit':
                dist = row['radius'] * dist_params[mode]['geodesic_distance'] / 2
                df.loc[key, 'dist_{}'.format(mode)] = dist
            else:
                for col in dist_cols:
                    name = col.replace('dist_', '')
                    dist = row['radius'] * dist_params[name]['geodesic_distance'] / 2
                    df.loc[key, col] = dist

# Convert distances to kilometers.
df['dist_walking'] /= 1000
df['dist_cycling'] /= 1000
df['dist_motorcycle'] /= 1000
df['dist_car'] /= 1000
for col in dist_cols:
    df[col] /= 1000

# Compute CO2 emissions of the trips.
kg_co2_per_km_car = 0.193
kg_co2_per_km_motorcycle = 0.0644
kg_co2_per_km_bus = 0.129 # Autobus agglomération > 250k habitants.
kg_co2_per_km_rail = 0.0248 # Regional Express Train (TER).
kg_co2_per_km_metro = 2.98e-3 # Tramway ou métro.
df['kg_co2_motorcycle'] = df['dist_motorcycle'] * kg_co2_per_km_motorcycle
df['kg_co2_car'] = df['dist_car'] * kg_co2_per_km_car
df['kg_co2_rail'] = df['dist_railRegional'] * kg_co2_per_km_rail
df['kg_co2_metro'] = df['dist_railMetro'] * kg_co2_per_km_metro
df['kg_co2_bus'] = df['dist_busPublic'] * kg_co2_per_km_bus
df['kg_co2_tram'] = df['dist_railLight'] * kg_co2_per_km_metro
df['kg_co2_public_transit'] = (
    df['kg_co2_bus'] + df['kg_co2_rail']
    + df['kg_co2_tram'] + df['kg_co2_metro']
)

df.to_csv('data/od_matrix_rhone.csv', index=False)
