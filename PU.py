import sklearn
import sklearn.base
import sklearn.preprocessing
import sklearn.svm
import sklearn.cross_validation
import sklearn.grid_search

import numpy as np

class PositiveUnlabeled(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    '''Positive-unlabeled learning.


    :parameters:

        - estimator : sklearn.base.ClassifierMixin
          A binary classifier.  Must support the ``predict_proba`` method.

        - sample_estimator : str
          One of:
            - mean : arithmetic mean of probabilities
            - gmean : arithmetic mean of log probabilities
            - max : max probability
            - ucb : upper confidence bound : mean + 3*stdev
            - gucb : ucb in log space
    '''
    def __mean(self, p):
        return np.mean(p)

    def __gmean(self, p):
        return np.exp(self.__mean(np.log(p)))

    def __ucb(self, p):
        m = np.mean(p)
        s = np.std(p)
        return m + 3*s

    def __gucb(self, p):
        return np.exp(self.__ucb(np.log(p)))

    def __max(self, p):
        return np.max(p)


    def __init__(self, estimator, sample_estimator='mean'):

        __ests = {'mean': self.__mean,
                  'gmean': self.__gmean,
                  'max': self.__max,
                  'ucb': self.__ucb,
                  'gucb': self.__gucb}

        self.estimator = estimator
        self.sample_estimator = sample_estimator
        assert hasattr(self.estimator, 'predict_proba')
        assert self.sample_estimator in __ests

    def fit(self, X, Y):
        __ests = {'mean': self.__mean,
                  'gmean': self.__gmean,
                  'max': self.__max,
                  'ucb': self.__ucb,
                  'gucb': self.__gucb}

        # Fit the estimator
        self.estimator.fit(X, Y)

        # Estimate the probability calibration
        idx = np.argwhere(Y > 0).ravel()

        positive_probabilities = self.estimator.predict_proba(X[idx])[:, 1]

        self.calibration_ = __ests[self.sample_estimator](positive_probabilities)

        self.log_calibration_ = np.log(self.calibration_)


    def predict_proba(self, X):

        # A brutal hack here to clip our scaled probability estimates 
        # into the feasible range
        pos_proba = np.minimum(1.0 - np.finfo(float).eps,
                               self.estimator.predict_proba(X)[:, 1] / self.calibration_)

        neg_proba = 1.0 - pos_proba

        return np.vstack([neg_proba, pos_proba]).T


    def predict_log_proba(self, X):

        return np.log(self.predict_proba(X))


    def decision_function(self, X):

        log_proba = self.predict_log_proba(X)
        return log_proba[:, 1] - np.log(0.5)


    def predict(self, X):
        probs = self.predict_proba(X)

        return (probs[:, 1] > probs[:, 0]).astype(int)

class PUSVM(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    '''Positive-unlabeled learning with one-class SVM calibration.


    :parameters:

        - estimator : sklearn.base.ClassifierMixin
          A binary classifier.  Must support the ``predict_proba`` method.

        - parameters : dict 
          A parameter dictionary for the one-class SVM
    '''

    def __init__(self, estimator, parameters=None):

        self.estimator = estimator
        self.parameters = parameters

        assert hasattr(self.estimator, 'predict_proba')

    def __calibrate(self, X, Y, pos_idx):

        # First, estimate the proportion of labeled points
        Ps = np.mean(Y, axis=0)

        # Instantiate a one-class svm to the positive data
        self.oner_ = sklearn.grid_search.GridSearchCV(sklearn.svm.OneClassSVM(), 
                                                      self.parameters, 
                                                      scoring='recall')
        # Fit the one-class SVM
        self.oner_.fit(X[pos_idx], Y[pos_idx])
        Ypred = (self.oner_.predict(X) > 0).astype(int)

        Py = np.mean(Ypred, axis=0)

        return Ps, Py

    def fit(self, X, Y):

        # Fit the estimator
        self.estimator.fit(X, Y)

        # Estimate the probability calibration
        pos_idx = np.argwhere(Y > 0).ravel()

        # FIXME:  2014-08-13 11:35:26 by Brian McFee <brm2132@columbia.edu>
        # calibration_ should be P(S=1) / P(Y=1)
        # = np.mean(Y) / np.mean(Ypred)

        Ps, Py = self.__calibrate(X, Y, pos_idx)

        self.calibration_ = float(Ps) / Py

        self.log_calibration_ = np.log(self.calibration_)


    def predict_proba(self, X):

        # A brutal hack here to clip our scaled probability estimates 
        # into the feasible range
        pos_proba = np.minimum(1.0 - np.finfo(float).eps,
                               self.estimator.predict_proba(X)[:, 1] / self.calibration_)

        neg_proba = 1.0 - pos_proba

        return np.vstack([neg_proba, pos_proba]).T


    def predict_log_proba(self, X):

        return np.log(self.predict_proba(X))


    def decision_function(self, X):

        log_proba = self.predict_log_proba(X)
        return log_proba[:, 1] - np.log(0.5)


    def predict(self, X):
        probs = self.predict_proba(X)

        return (probs[:, 1] > probs[:, 0]).astype(int)
