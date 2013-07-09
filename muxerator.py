#!/usr/bin/env python
"""Randomized generator multiplexer
"""

import random

class muxerator(object):
    """Randomized generator multiplexer class."""
    def __init__(self, pool):
        """Constructor.  Given a list of generators, the muxerator will generate data
        in a similar fashion to `itertools.chain`, but rather than going sequentially
        from one generator to the next, will randomly select a different one each time.

        :parameters:
        - pool      : list of generators
            The set of generators to multiplex over

        """

        self.pool = pool

    def __iter__(self):
        return self

    def next(self):

        while len(self.pool) > 0:
            i = random.randint(0, len(self.pool)-1)

            try:
                return self.pool[i].next()
            except StopIteration:
                self.pool.pop(i)
        raise StopIteration
