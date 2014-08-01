# CREATED:2014-08-01 16:36:41 by Brian McFee <brm2132@columbia.edu>
# Cross-validation and shuffle splitting with grouped meta-identifiers 

import sklearn.cross_validation
import sklearn.preprocessing
import collections

class MetaShuffleSplit(sklearn.cross_validation.BaseShuffleSplit):
    '''Analogous to ShuffleSplit, except partitioning happens over
    the range of a many-to-one mapping of the dataset.

    This is useful for tying certain datapoints to the same side of a split.
    For example, if the data are a collection of n songs by m unique artists,
    it is common to require that no artist exist on both sides of the split.

    :parameters:

    - meta_ids : list of hashable
      A list of identifiers for the data.  ids can be any hashable type (eg, strings)

    - n_iter : int > 0
      number of splits to generate

    - test_size : float > 0
      fraction of points in the test set

    - train size : float > 0 or None
      fraction of points in the training set (optional)

    - random state : seed
      seed for the PRNG

    .. seealso:: sklearn.cross_validation.ShuffleSplit
    '''
    def __init__(self, meta_ids, n_iter=10, test_size=0.1, train_size=None, random_state=None):

        self.n = len(meta_ids)
        self.meta_ids = meta_ids

        self.n_iter = n_iter
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._indices = True

    def _iter_indices(self):

        # 1. collapse the map by label encoding
        id_mapping = collections.defaultdict(list)

        self.meta_ids_encoded_ = sklearn.preprocessing.LabelEncoder().fit_transform(self.meta_ids)

        for idx, meta in enumerate(self.meta_ids_encoded_):
            id_mapping[meta].append(idx)

        # 2. Instantiate the shufflesplitter
        n = len(id_mapping)

        splitter = sklearn.cross_validation.ShuffleSplit(n,
                                                         n_iter=self.n_iter,
                                                         test_size=self.test_size,
                                                         train_size=self.train_size,
                                                         random_state=self.random_state)

        # 3. Yield iterates
        for raw_train, raw_test in splitter:
            train = []
            test = []

            for raw in raw_train:
                train.extend(id_mapping[raw])
            for raw in raw_test:
                test.extend(id_mapping[raw])

            yield train, test

    def __repr__(self):
        return ('%s(meta_ids=%s, n_iter=%d, test_size=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.meta_ids,
                    self.n_iter,
                    str(self.test_size),
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter

