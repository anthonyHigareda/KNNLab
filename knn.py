import random
from operator import indexOf

import numpy as np
from scipy.spatial import distance
from scipy import stats
import math

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification.
    """
    def __init__(self, k):
        """
        Takes one parameter.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point.
        """
        self.k = k
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        """
        self.X = X
        self.y = y
        
    def predict_loop(self, X):
        """
        Predicts the output variable's values for the query points X using loops.
        
        """
        solved_types = []
        for j in range (len(X)):
            distances = []
            y_types = []
            y_types_hits = []
            for i in range (len(self.X)): # get the point's distance from each know point
                # add the distance and classification type as a tuple to the distances list
                distances.append((distance.euclidean(self.X[i], X[j]), self.y[i]))
                # add only new types to the y_types list
                if self.y[i] not in y_types:
                    y_types.append(self.y[i])
                    y_types_hits.append(0) # set the number of hits variable for each y_type to 0
            distances = sorted(distances, key=lambda dist: dist[0]) # sort distances from least to greatest

            for i in range (self.k): # look at each of the closest neighbors and note the y_type
                y_types_hits[indexOf(y_types, distances[i][1])] += 1

            most_likely_type = y_types[0]
            max_hits = y_types_hits[0]
            for i in range (len(y_types_hits)):
                if y_types_hits[i] > max_hits:
                    most_likely_type = y_types[i]
                    max_hits = y_types_hits[i]

            solved_types.append(most_likely_type)

        return np.asarray(solved_types)