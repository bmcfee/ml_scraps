#!/usr/bin/env python
# CREATED:2013-05-05 11:14:34 by Brian McFee <brm2132@columbia.edu>
# sklearn.decomposition container class for vector quantization 

import numpy as np
import sklearn.cluster
from sklearn.base import BaseEstimator, TransformerMixin

class VectorQuantizer(BaseEstimator, TransformerMixin):

    def __init__(self, clusterer=None, n_atoms=32):
        '''Vector quantization by closest centroid:

        A[i] == 1 <=> i = argmin ||X - C_i||
                        i

        Arguments:
        ----------
        n_atoms : int
            Number of dictionary elements to extract
    
        clusterer : {None, BaseEstimator}
            Instantiation of a clustering object (eg. sklearn.cluster.MiniBatchKMeans)

            default: sklearn.cluster.MiniBatchKMeans

        n_atoms : int
            If no clusterer is provided, the number of atoms to use
        '''

        if clusterer is None:
            self.clusterer = sklearn.cluster.MiniBatchKMeans(k=n_atoms)
        else:
            self.clusterer = clusterer

    def fit(self, X):
        '''Fit the codebook to the data

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data

        Returns
        -------
        self : object
        '''
        
        self.clusterer.fit(X)
        C = self.clusterer.cluster_centers_
        self.center_norms_ = 0.5 * np.diag(np.dot(C, C.T))
        return self

    def transform(self, X):
        '''Encode the data by VQ.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data to be transformed

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms)
        '''

        C = self.clusterer.cluster_centers_

        XC = np.dot(X, C.T) - self.center_norms_

        X_new = np.zeros( (X.shape[0], C.shape[0]), dtype=bool )
        
        hits = XC.argmax(axis=1)
        for i in range(X.shape[0]):
            X_new[i, hits[i]] = True

        return X_new
