"""Description of census data.

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""
import os

import numpy as np
import pandas as pd
import geopandas as gpd

#################
#  Data Import  #
#################

# Set current directory to file directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Open the census data.
# https://www.insee.fr/fr/statistiques/4507890
df = pd.read_csv(
    'data/FD_MOBPRO_2017.csv',
    sep=';',
    usecols=[
        'COMMUNE', 'ARM', 'DCLT', 'AGEREVQ', 'CS1', 'DIPL', 'EMPL', 'ILT',
        'INPOM', 'IPONDI', 'METRODOM', 'NPERR', 'SEXE', 'TRANS', 'VOIT',
    ],
    na_values=['Z', 'ZZZZZ'],
    dtype={'COMMUNE': str, 'ARM': str, 'DCLT': str, 'DIPL': str},
)

# Open French city geographic data.
# https://www.data.gouv.fr/fr/datasets/decoupage-administratif-communal-francais-issu-d-openstreetmap/
com_df = gpd.read_file('data/communes.shp')
com_df = com_df.drop(columns='wikipedia')
arr_df = gpd.read_file('data/arrondissements-municipaux.shp')
arr_df = arr_df.drop(columns=['wikipedia', 'type', 'start_date', 'end_date'])
geo_df = com_df.append(arr_df)

#######################
#  Global Statistics  #
#######################

nb_households = df.shape[0]
print('Total number of households: {}'.format(nb_households))

nb_individuals_total = int(df['NPERR'].sum())
print('Total number of individuals: {}'.format(nb_individuals_total))


###################
#  Data Cleaning  #
###################

# Keep only individuals living in Metropolitan France.
df = df.loc[df['METRODOM']=='M']
df = df.drop(columns='METRODOM')
# Keep only individuals with specified workplace.
df = df.loc[df['DCLT']!=99999]
# Remove individuals not working in Metropolitan France.
df = df.loc[~df['ILT'].isin([5, 6, 7])]
# Remove non-commuting individuals.
df = df.loc[df['TRANS']!=1]
# Drop NAs.
df = df[~df['VOIT'].isna()]

############################
#  Home and Work Location  #
############################

# Set the home city or district.
df['home'] = df['COMMUNE']
df.loc[~df['ARM'].isna(), 'home'] = df['ARM']
# Set the workplace city or district.
df['work'] = df['DCLT']
df = df.drop(columns=['COMMUNE', 'ARM', 'DCLT'])

# Keep only individuals living in the Rhone department.
df = df.loc[(df['home'].str.startswith('69'))&(df['work'].str.startswith('69'))]

nb_individuals = df.shape[0]
print('Number of individuals (after data cleaning): {}'.format(nb_individuals))

# Compute the OD matrix.
od_matrix_size = df.groupby(['home', 'work']).size()
od_matrix_size.name = 'size'
od_matrix_weight = df.groupby(['home', 'work'])['IPONDI'].sum()
od_matrix_weight.name = 'weight'
od_matrix = pd.merge(od_matrix_size, od_matrix_weight,
                     left_index=True, right_index=True)
od_matrix = od_matrix.reset_index()
od_matrix = od_matrix.merge(
    geo_df[['insee', 'surf_ha']],
    left_on='home',
    right_on='insee',
    how='left',
)
if np.any(od_matrix['insee'].isna()):
    invalid_codes = od_matrix.loc[od_matrix['insee'].isna(), 'home'].values
    print('Warning. Unrecognized INSEE codes: {}'.format(invalid_codes))
od_matrix = od_matrix.drop(columns='insee')
av_surf_ha = (
    (od_matrix['surf_ha'] * od_matrix['weight']).sum()
    / od_matrix['weight'].sum()
)
# Convert from hectare to square kilometers.
av_surf = av_surf_ha / 100

nb_uniques_home = od_matrix['home'].nunique()
print('Number of uniques home location: {}'.format(nb_uniques_home))
av_indiv_per_home = nb_individuals / nb_uniques_home
print('Average number of individual per home location: {}'.format(av_indiv_per_home))
nb_uniques_work = od_matrix['work'].nunique()
print('Number of uniques work location: {}'.format(nb_uniques_work))
av_indiv_per_work = nb_individuals / nb_uniques_work
print('Average number of individual per work location: {}'.format(av_indiv_per_work))
print('Average surface of home location (in km2): {}'.format(av_surf))

# Save the OD matrix.
od_matrix.to_csv('data/od_matrix_rhone.csv', index=False)


############################
#  Mode of Transportation  #
############################

df['TRANS'] = df['TRANS'].astype('category')
df['TRANS'] = df['TRANS'].cat.rename_categories(
    ['walking', 'cycling', 'motorcycle', 'car', 'public transit']
)
shares = df.groupby('TRANS')['IPONDI'].sum() / df['IPONDI'].sum()
print('Share of each mode of transportation:')
print(shares)


##########################
#  Number of Cars Owned  #
##########################

df['VOIT'] = df['VOIT'].astype(np.int64)
# Number of cars per employed.
df['car_per_indiv'] = df['VOIT'] / df['INPOM']
shares = df.groupby('VOIT')['IPONDI'].sum() / df['IPONDI'].sum()
print('Share of car ownership:')
print(shares)
av_nb_cars = (df['car_per_indiv'] * df['IPONDI']).sum() / df['IPONDI'].sum()
print('Average number of cars owned per individual: {}'.format(av_nb_cars))


#################################
#  Socio-Demographic Variables  #
#################################

av_age = (df['AGEREVQ'] * df['IPONDI']).sum() / df['IPONDI'].sum()
print('Average age: {}'.format(av_age))

df['CS1'] = df['CS1'].astype('category')
df['CS1'] = df['CS1'].cat.rename_categories(
    ['Farmer', 'Artisan', 'Executive', 'Intermediate', 'Employee',
     'Blue-collar']
)
shares = df.groupby('CS1')['IPONDI'].sum() / df['IPONDI'].sum()
print('Share of professions:')
print(shares)

df['EMPL'] = df['EMPL'].astype('category')
df['EMPL'] = df['EMPL'].cat.rename_categories(
    ['Apprenticeship', 'Interim', 'Subsidized employment', 'Internship',
     'Fixed-term', 'Permanent', 'Independent', 'Employer', 'Family assistance']
)
shares = df.groupby('EMPL')['IPONDI'].sum() / df['IPONDI'].sum()
print('Share of contracts:')
print(shares)

dipl_dict = {
    '01': 'No degree', '02': 'No degree', '03': 'No degree',
    '11': 'Primary school', '12': 'Middle school', '13': 'Secondary',
    '14': 'Secondary', '15': 'Secondary', '16': 'Undegraduate',
    '17': 'Undegraduate', '18': 'Graduate', '19': 'PhD',
}
df['DIPL'] = df['DIPL'].map(dipl_dict).astype('category')
shares = df.groupby('DIPL')['IPONDI'].sum() / df['IPONDI'].sum()
print('Share of degrees:')
print(shares)

df['SEXE'] = df['SEXE'].astype('category')
df['SEXE'] = df['SEXE'].cat.rename_categories(
    ['Homme', 'Femme']
)
shares = df.groupby('SEXE')['IPONDI'].sum() / df['IPONDI'].sum()
print('Share of each sex:')
print(shares)

# Save the cleaned census data.
df.to_csv('data/population_rhone.csv', index=False)
