#!/usr/bin/env python
# CREATED:2013-06-08 09:33:43 by Brian McFee <brm2132@columbia.edu>
#  random feature hashing and hash cascades

import numpy as np
import scipy.stats
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin

class RandomHash(BaseEstimator, TransformerMixin):

    def __init__(self, n_atoms=64, quantile=0.5, projection='gaussian'):
        '''Random hash transformation.

        Given d-dimensional input data, produces a n_atoms-dimensional binary vector
        by thresholding a random projection of the data.

        Arguments
        ---------
        n_atoms : int
            The number of hash dimensions

        quantile : float (0,1)
            Quantile threshold for output features. Defaults to 0.5 for median.

        projection : str {'gaussian', 'rademacher', 'boolean'}
            What type of random projection matrix to use
        '''

        self.n_atoms    = n_atoms
        self.quantile   = quantile
        self.projection = projection


    def fit(self, X):
        '''Fit the random hash

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]

        Returns
        -------
        self : object
        '''

        n, d = X.shape

        # Generate the projection 
        W = np.random.randn(d, self.n_atoms)

        if self.projection == 'rademacher':
            W = np.sign(W).astype(np.float)
        elif self.projection  == 'boolean':
            W = (W > 0).astype(np.float)

        # Fit the thresholds
        self.components_ = W

        thresholds = np.zeros(self.n_atoms)

        for i in range(self.n_atoms):

            thresholds[i] = np.array(
                            scipy.stats.mstats.mquantiles(X.dot(W[:,i]), 
                                                        prob=[self.quantile],
                                                        axis=0))
        self.thresholds_ = thresholds

        return self

    def transform(self, X):
        '''Transform data by the random hash

        Parameters
        ----------
        X : arary-like, shape [n_samples, n_features]
            Data to be transformed

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms)
            Binary output matrix
        '''

        return X.dot(self.components_) >= self.thresholds_

class RandomHashCascade(BaseEstimator, TransformerMixin):

    def __init__(self, n_layers=3, n_atoms=64, quantile=0.5, projection='gaussian', sparse=False):
        '''Random hash cascade

        Parameters
        ----------
        n_layers : int
            Number of layers in the cascade

        n_atoms : int or list-like
            Number of atoms at each layer

        quantile : float or list-like
            Quantile-threshold for each layer in the cascade

        projection : str or list-like
            Projection form for each layer in the cascade

        sparse : boolean
            Sparsify the output?
        '''

        self.n_layers   = n_layers
    
        # Expand out parameters
        def _listify(x):
            if not hasattr(x, '__iter__'):
                x = [x] * self.n_layers
            return x
    
        self.n_atoms    = _listify(n_atoms)
        self.quantile   = _listify(quantile)
        self.projection = _listify(projection)
        self.sparse     = sparse

    def fit(self, X):
        '''Fit the random hash cascade

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        self : object
        '''

        X_new = X
        self.hashes_ = []

        for i in range(self.n_layers):
            H = RandomHash( n_atoms     =   self.n_atoms[i],
                            quantile    =   self.quantile[i],
                            projection  =   self.projection[i] )

            X_new = H.fit_transform(X_new)
            self.hashes_.append(H)

        return self

    def transform(self, X):
        '''Transform data through the random hash cascade

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be transformed

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms[-1])
            Transformed data
        '''

        X_new = X

        for H in self.hashes_:
            X_new = H.transform(X_new)

        if self.sparse:
            X_new = scipy.sparse.csr_matrix(X_new)

        return X_new
