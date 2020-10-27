"""
Simple implementation of DBSCAN clustering intended for learning purposes.

use this code however you like.
"""


import numpy as np
import math

class dbscan():
    
    def __init__(self, eps=0.5, min_points=5):
        self.eps = eps
        self.min_points = min_points
        self.noise = -1
        self.unclassified = 0
        
    def fit(self, X):
        """
        fit the data
        """

        cluster = 1
        n_points = X.shape[1]
        labels = [self.unclassified] * n_points

        for index in range(0, n_points):
            point = X[:, index]
            if labels[index] == self.unclassified:
                if self.grow_cluster(X, index, labels, cluster):
                    cluster = cluster + 1
        self.labels = labels
        return self
        
        
    def predict(self, X):
        """
        return a list of cluster labels. The label -1 means noise, and then
        the clusters are numbered starting from 1
        """

        self.fit(X)
        return self.labels
    
    
    def grow_cluster(self, X, index, labels, cluster):
        """
        Function checks if a new cluster can be grown from each seed point then
        labels them as either a noise point or assigns the proper cluster number
        """
        seeds = self.region_query(X, index)
        if len(seeds) < self.min_points:
            labels[index] = self.noise
            return False
        else:
            labels[index] = cluster
            for seed in seeds:
                labels[seed] = cluster
                
            while len(seeds) > 0:
                position = seeds[0]
                results = self.region_query(X, position)
                if len(results) >= self.min_points:
                    for i in range(0, len(results)):
                        result_position = results[i]
                        if labels[result_position] == self.unclassified or \
                            labels[result_position] == self.noise:
                            if labels[result_position] == self.unclassified:
                                seeds.append(result_position)
                            labels[result_position] = cluster
                seeds = seeds[1:]
            return True
        
        
    def region_query(self, X, index):
        """
        Function calculates the distance between a point against every
        other point in the dataset and returns all points that are within
        the epsilon threshhold
        """

        n_points = X.shape[1]
        neighbors = []
        for i in range(0, n_points):
            if self.eps_neighborhood(X[:, index], X[:, i]):
                neighbors.append(i)
        return neighbors
    
    def eps_neighborhood(self, p, q):
        """
        Checks if distance is less than epsilon
        """

        return self.dist(p, q) < self.eps
    
    def dist(self, p, q):
        """
        Euclidian distance function
        """

        return math.sqrt(np.power(p-q, 2).sum())
