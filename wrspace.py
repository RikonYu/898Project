from gym.spaces import Space
import numpy as np  # takes about 300-400ms to import, so we load lazily
class TraceSpace(Space):
    def __init__(self, n,):
        self.n = n
        super(Discrete, self).__init__((), np.int64)
