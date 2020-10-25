import pandas as pd
import numpy as np
import random as rd
from scipy.spatial import distance

from algorithms.KMeans import KMeans

class KMedians(KMeans):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Distance function - Manhattan Distance Function
        self.metric = 'cityblock'

    def _update_centroids(self):
        """Compute the new centroid for each cluster using method depending on algorithm."""
        for k in range(self.k):
            if len(self.nearest[k]) > 0:
                self.centroids[k, :] = np.median(np.array(self.nearest[k]), axis=0)