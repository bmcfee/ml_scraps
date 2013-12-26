#!/usr/bin/env python
# CREATED:2013-12-11 17:26:05 by Brian McFee <brm2132@columbia.edu>
# sklearn module for whitening transformations 


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Whitening(BaseEstimator, TransformerMixin):

    def __init__(self):
        '''Whitening transformation:

        X_white = (X_white - mu) / std

        '''
        self.count   = 0
        pass


    def fit(self, X):
        '''Estimate whitening parameters

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data

        Returns
        -------
        self : object
        '''
        
        # Initialize model parameters
        self.sum    = np.zeros( (1, X.shape[1]) )
        self.sumsq  = np.zeros_like(self.sum)
        self.count  = 0

        self.partial_fit(X)

        return self

    def partial_fit(self, X):
        if not hasattr(self, 'sum'):
            self.sum    = np.zeros( (1, X.shape[1]) )
            self.sumsq  = np.zeros_like(self.sum)
            self.count  = 0

        self.sum    += np.sum(X, axis=0, keepdims=True)
        self.sumsq  += np.sum(X**2, axis=0, keepdims=True)
        self.count  += X.shape[0]

        self.mean_  = self.sum / self.count
        self.std_   = np.sqrt(self.sumsq/self.count - self.mean_**2)
        return self


    def transform(self, X):
        '''Apply the estimated whitening transformation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data to be transformed

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms)
        '''

        if self.count < 2:
            raise Exception('Insufficient data to estimate variance')

        return (X - self.mean_) / (self.std_ + (self.std_ == 0))
