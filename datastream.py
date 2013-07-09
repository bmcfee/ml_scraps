#!/usr/bin/env python
"""Data stream generator.


"""
import muxerator

class datastream(object):
    """Data stream multiplexer."""
    def __init__(self, mapper, objects, k=10):
        """

        :example:
        ```
        def mapper(obj):
            for z in np.load(obj):
                yield z

        G = datastream(mapper, glob.glob('file_*.npy'), k=5)
        ```
        The `mapper` function generates data by loading from the `obj` file.
        `G` in turn maintains a live set of `k=5` generators according to the `objects`
        glob, and generates data by randomly selecting among them until all data is
        exhausted.

        See `muxerator` for details.

        :parameters:
        - mapper : function
            Takes an object from `objects` and returns a generator
        - objects : list of objects
            Each object will be passed into `mapper` to construct a generator
        - k : int>0
            Maximum size of the active set.
        """

        self.k          = k
        self.mapper     = mapper
        self.objects    = objects

    def __iter__(self):
        self.restock()
        return self

    def restock(self):
        pool = []

        if len(self.objects) == 0:
            raise StopIteration

        while len(self.objects) > 0 and len(pool) < self.k:
            pool.append(self.mapper(self.objects.pop(0)))

        self.generator = muxerator.muxerator(pool)

        pass

    def next(self):
        # randomly select a generator from

        while True:
            try:
                return self.generator.next()
            except StopIteration:
                pass
            self.restock()
