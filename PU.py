import sklearn
import sklearn.base
import sklearn.preprocessing
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


    def __init__(self, estimator, sample_estimator='gmean'):

        ests = {'mean': self.__mean,
                'gmean': self.__gmean,
                'max': self.__max,
                'ucb': self.__ucb,
                'gucb': self.__gucb}

        assert hasattr(estimator, 'predict_proba')
        assert sample_estimator in ests

        self.estimator_ = estimator
        self.sample_estimator_ = sample_estimator
        self.sample_estimator_function_ = ests[sample_estimator]


    def fit(self, X, Y):
        # Fit the estimator
        self.estimator_.fit(X, Y)

        # Estimate the probability calibration
        idx = np.argwhere(Y > 0).ravel()

        positive_probabilities = self.estimator_.predict_proba(X[idx])[:, 1]

        self.calibration_ = self.sample_estimator_function_(positive_probabilities)
        #probabilities = self.estimator_.predict_proba(X)[:, 1]

        #self.calibration_ = np.sum(probabilities * Y) / np.sum(probabilities)

        self.log_calibration_ = np.log(self.calibration_)


    def predict_proba(self, X):

        # A brutal hack here to clip our scaled probability estimates into the feasible range
        pos_proba = np.minimum(1.0 - np.finfo(float).eps,
                               self.estimator_.predict_proba(X)[:, 1] / self.calibration_)

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
