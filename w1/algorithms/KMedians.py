import pandas as pd
import numpy as np
import random as rd
from scipy.spatial import distance

class KMedians(KMeans):
    def __init__(self, k:int, max_it=100, seed=1, tol=1e-4):
    	KMeans.__init__(self, k, max_it, seed, tol)
    	# Distance function - Manhattan Distance Function
        self.metric = 'cityblock'
        # Allows to choose the new centroid of the position of the median vector.
        self.cost_function_type = 'median'
        