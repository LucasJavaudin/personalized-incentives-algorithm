"""Python script used to generate Figure 2 from the paper.

Author: Lucas Javaudin
E-mail: lucas.javaudin@cyu.fr
"""

import os

import algorithm

OUTPUT_DIR = ''

algorithm.distance_optimum(
    individuals=10,
    mean_nb_alternatives=5,
    use_poisson=False,
    filename=os.path.join(OUTPUT_DIR, 'distance_optimum.pdf'),
    bounds=True,
    title=None,
    seed=0,
)
