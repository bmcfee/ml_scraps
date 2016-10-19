#!/usr/bin/env python
# CREATED:2013-11-18 17:25:11 by Brian McFee <brm2132@columbia.edu>
# sklearn.decomposition container for multi-label vector quantization

import numpy as np
import scipy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from VectorQuantizer import VectorQuantizer

class MultiVQ(BaseEstimator, TransformerMixin):
    def __init__(self,  n_atoms=32, 
                        n_classes=2, 
                        sparse=True, 
                        batch_size=1024):
        '''Multi-class vector quantization
        '''

        self.vqs = [VectorQuantizer(n_atoms=n_atoms, 
                                    sparse=True,
                                    batch_size=batch_size) for _ in range(n_classes)]

        self.sparse     = sparse
        self.batch_size = batch_size
        
    def fit(self, X, Y):
        ''' '''

        # Make sure Y is binary-friendly
        L = LabelBinarizer()
        Y = L.fit_transform(Y)

        for C in range(len(self.vqs)):
            # Get the subset with this label
            idx = (Y[:, C] > 0).flatten()
            if idx.size > 0:
                self.vqs[C].fit(X[idx])

        return self

    def partial_fit(self, X, Y):
        L = LabelBinarizer()
        Y = L.fit_transform(Y)

        for C in range(len(self.vqs)):
            idx = (Y[:, C] > 0).flatten()
            if idx.size > 0:
                self.vqs[C].partial_fit(X[idx])

        return self

    def transform(self, X):
        if self.sparse:
            return scipy.sparse.hstack([C.transform(X) for C in self.vqs])
        else:
            return np.hstack([C.transform(X) for C in self.vqs])
