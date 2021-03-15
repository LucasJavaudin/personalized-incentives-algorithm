# Application to Mode Choice in France

## Requirements

- All Python packages defined in `requirements.txt` must be installed.
- The file `data/FD_MOBPRO_2017.csv` must contain the census data from [INSEE](https://www.insee.fr/fr/statistiques/4507890).
- The files `data/communes.shp` and `data/arrondissements-municipaux.shp` must contain the geometries of the French _communes_ and _arrondissements_ from [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/decoupage-administratif-communal-francais-issu-d-openstreetmap/).
- The variable `HERE_API_KEY` for the script `03_compute_travel_times.py` must be set to a valid [HERE API](https://developer.here.com/) key.
- The variable `wd` in `05_regressions.R` must be set to the absolute path of the directory `application/`.
- The url `http://localhost:8989/route` must point to a valid [GraphHopper](https://www.graphhopper.com/) server that can supply shortest paths by foot, bike and motorcycle for the [Rhone-Alpes area](http://download.geofabrik.de/europe/france/rhone-alpes.html).

## How-to Run

All scripts must be run in order.
- `01_data_description.py`: this script computes a few descriptive statistics from the census data.
- `02_retrieve_townhalls.py`: this script retrieve the townhalls location of French cities form Overpass API.
- `03_compute_travel_times.py`: this script compute travel times for all modes of transportation, for all individuals, from HERE API and GraphHopper.
- `04_prepare_regressions.py`: this script merge census data with travel-times data.
- `05_regressions.R`: this script runs the multinomial-logit regression in R.
- `06_prepare_run.py`: this script use the results of the regression and the CO2 emissions to build an input data for the algorithm.
- `07_algorithm.py`: this script runs the MCKP algorithm and generate some results and plots.
- `08_imp_info.py`: this script runs the MCKP algorithm with imperfect information.
- (optional) `09_solver.py`: this script runs a solver for the MCKP problem, to compare running time with the greedy algorithm.
- (optional) `regression_stats.py`: this script computes some statistics from the regression results.
