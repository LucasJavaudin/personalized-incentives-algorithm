"""Retrieve townhalls of all French cities using Overpass API.

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import json

import requests
import pandas as pd

# Retrieve the administrative centre of the cities.

overpass_url = 'http://overpass-api.de/api/interpreter'
#overpass_url = 'https://overpass.kumi.systems/api/interpreter'
overpass_query = f"""
[out:json][timeout:600];
area["name"="France"]->.searchArea;
relation["admin_level"="8"](area.searchArea)->.relations;
(node(r.relations:"admin_centre"););
.relations out;
out body;
>;
out skel qt;
"""
response = requests.get(overpass_url, params={'data': overpass_query})

data = json.loads(response.content)

# Read the data.
admin_centre = dict()
nodes = dict()
for element in data['elements']:
    try:
        if element['type'] == 'relation':
            for node in element['members']:
                if node['role'] == 'admin_centre':
                    admin_centre[element['tags']['ref:INSEE']] = node['ref']
                    break
        elif element['type'] == 'node':
            nodes[element['id']] = element['lat'], element['lon']
    except KeyError:
        pass

# Retrieve the administrative centre of the districts.

overpass_url = 'http://overpass-api.de/api/interpreter'
#overpass_url = 'https://overpass.kumi.systems/api/interpreter'
overpass_query = f"""
[out:json][timeout:600];
area["name"="France"]->.searchArea;
relation["admin_level"="9"]["ref:INSEE"](area.searchArea)->.relations;
(node(r.relations:"admin_centre"););
.relations out;
out body;
>;
out skel qt;
"""
response = requests.get(overpass_url, params={'data': overpass_query})

data = json.loads(response.content)

# Read the data.
for element in data['elements']:
    try:
        if element['type'] == 'relation':
            for node in element['members']:
                if node['role'] == 'admin_centre':
                    admin_centre[element['tags']['ref:INSEE']] = node['ref']
                    break
        elif element['type'] == 'node':
            nodes[element['id']] = element['lat'], element['lon']
    except KeyError:
        pass

# Create a DataFrame.
df = pd.DataFrame(columns=['insee', 'lat', 'lon'])
for key, value in admin_centre.items():
    lat, lon = nodes[value]
    df = df.append(dict(
        insee=key,
        lat=lat,
        lon=lon,
    ), ignore_index=True)

# Save the GeoJson to a file.
df.to_csv('data/admin_centre.csv', index=False)
