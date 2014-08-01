import sklearn.base
import sklearn.preprocessing
import collections

class TagEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    '''A variation on the LabelEncoder module which supports extended tag
    vocabularies at test time, and iterables of tags at training/test time.

    :parameters:

    - discard_missing : bool
      If true, then new labels are discarded at test time

    - min_count : int > 0
      Minimum number of occurrences for a tag to be included in the encoder.
      By default, one occurrence is enough.

    .. seealso: sklearn.preprocessing.LabelEncoder


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

        # Only retain tags with enough occurrences
        for t in all_tags:
            if self.vocab_[t] < self.min_count:
                del self.vocab_[t]

        # Build and fit a label encoder
        self.encoder_ = sklearn.preprocessing.LabelEncoder()
        self.encoder_.fit(sorted(self.vocab_.keys()))
        return self

    def transform(self, Y):
        '''Apply a tag transformation

        :parameters:

        - Y : iterable of iterable of hashables
          As in fit()

        :returns:

        - Yout : list
          Yout[i] is a numpy array of encoded labels


        '''
        Yout = []

        for yi in Y:
            if self.discard_missing:
                yi = [t for t in yi if t in self.vocab_]

            Yout.append(self.encoder_.transform(yi))

        return Yout

    def inverse_transform(self, y):

        Yout = []

        for yi in y:
            Yout.append(set(self.encoder_.inverse_transform(yi)))
        return Yout
