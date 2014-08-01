# CREATED:2014-08-01 12:29:05 by Brian McFee <brm2132@columbia.edu>
# Multi-label filtering preprocessing
#   Allows you to remove rare classes prior to training
#   Also allows you to discard novel classes at test time

import sklearn.base
import collections

class MultiLabelFilter(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    '''A label processing filter that handles pruning of rare classes, and
    optionally suppressing novel classes at test time.
    '''

    def __init__(self, discard_missing=True, min_count=1):

        assert min_count > 0

        self.discard_missing = discard_missing
        self.min_count = min_count

    def fit(self, Y):
        '''Fit a tag encoder.

        :parameters:
        - Y : iterable of iterables of hashables
          Y[i] should be an iterable of hashable tag indicators (eg, a dictionary or a list of strings)

        '''

        # Initialize the vocabulary
        self.vocab_ = collections.defaultdict(int)

        # Count up occurrences in the training set
        for yi in Y:
            for tag in yi:
                self.vocab_[tag] += 1

        # Get a list of all unique tags
        all_tags = self.vocab_.keys()

        min_count = self.min_count

        if isinstance(self.min_count, float):
            min_count = int(min_count * len(Y))

        # Only retain tags with enough occurrences
        for t in all_tags:
            if self.vocab_[t] < min_count:
                del self.vocab_[t]

        return self

    def transform(self, Y):

        return [[t for t in yi if not self.discard_missing or t in self.vocab_] for yi in Y]
