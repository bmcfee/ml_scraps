#!/usr/bin/env python

import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator

class Hartigan(BaseEstimator):
    '''Hartigan clustering with optional graph constraints.

    repeat until convergence:
        for each point:
            find the cluster assignment with largest cost improvement

    When a constraint graph is specified, each point can only move to clusters
    containing one of its neighbors.

    '''

    def __init__(self, n_clusters=2, connectivity=None, max_iter=None, verbose=False):
        '''Initialize a Hartigan clusterer.

        :parameters:
        - n_clusters : int
            Number of clusters
    
        - connectivity : scipy.sparse.coo_matrix or None
            Connectivity graph ala sklearn.feature_extraction.image.grid_to_graph()
            If 'None', a fully connected (ie, unstructured) graph will  be generated.
    
        - max_iter : int or None
            Maximum number of passes through the data

        - verbose : bool
            Display debugging output?

        :variables:
        - labels_ : array
            Cluster assignments after fitting the model
    
        - components_ : model
            Estimated cluster centroids
        '''

        self.n_clusters     = n_clusters
        self.connectivity   = connectivity
        self.max_iter       = max_iter
        self.verbose        = verbose

    def fit(self, X, labels=None):
        '''Fit the cluster centers.

        :parameters:
        - X : ndarray, size=(n, d)
            The data to be clustered

        - labels : None, list-like size=n (int)
            Optional list of initial cluster assignments.
            Must be in the range [0, n_clusters-1]
            If unspecified, will be initialized randomly.

        :note:
            If using a connectivity graph, randomly initialized labels may not be
            consistent with the constraints.

        '''

        n, d = X.shape

        if self.connectivity is None:
            self.connectivity = scipy.sparse.coo_matrix(np.ones((n,n)))

        if labels is None:
            labels = np.random.randint(low=0, high=self.n_clusters, size=n)
        else:
            labels = np.array(labels, dtype=int).flatten()

        # Construct the constraint graph
        E = [set() for i in range(n)]
        for (i, j) in np.vstack(self.connectivity.nonzero()).T:
            E[i].add(j)
            E[j].add(i)
            
        # Initialize the cluster centers, costs, sizes
        self.components_    = np.zeros( (self.n_clusters, d), dtype=X.dtype)
        sizes               = np.zeros(self.n_clusters)

        for i in range(self.n_clusters):
            pts_i               = np.argwhere(labels == i).flatten()
            sizes[i]            = len(pts_i)
            if sizes[i] > 0:
                self.components_[i]     = np.mean(X[pts_i], axis=0)

        step = 0

        while self.max_iter is None or step < self.max_iter:
            step = step + 1

            n_changes = 0
            for i in range(n):
                l_old = labels[i]

                # if the cluster is a singleton, skip it
                if sizes[l_old] == 1:
                    continue

                # find the legal moves for point i
                moves = list(set([labels[j] for j in E[i]]) - set([l_old]))

                # if there are none, skip i
                if not moves:
                    continue
                
                # Otherwise, compute the cost-delta of moving to each candidate
                phi_old = sizes[l_old] * np.sum((self.components_[l_old] - X[i])**2) / (sizes[l_old] - 1.0)

                move_costs = []
                for l_new in moves:
                    phi_new = sizes[l_new] * np.sum((self.components_[l_new] - X[i])**2) / (sizes[l_new] + 1.0)
                    move_costs.append((phi_old - phi_new,  l_new))

                (cost_new, l_new) = max(move_costs)
                
                # If the best move has negative value, skip it
                if cost_new < 0:
                    continue

                # Else, re-assign X[i] and update the clustering
                self.components_[l_old] = (sizes[l_old] * self.components_[l_old] - X[i]) / (sizes[l_old] - 1.0)
                sizes[l_old]            = sizes[l_old] - 1.0


                self.components_[l_new] = (sizes[l_new] * self.components_[l_new] + X[i]) / (sizes[l_new] + 1.0)
                sizes[l_new]            = sizes[l_new] + 1.0

                labels[i]               = l_new
                
                # We had a change, so reset the convergence meter
                n_changes += 1

            if self.verbose:
                print 'Round %4d, %5d updates' % (step, n_changes)

            if n_changes == 0:
                break

        self.labels_ = labels
        self.sizes_  = sizes
        ###
