import numpy as np
from .strategy_otf import StrategyOTF

class RandomSamplingOTF(StrategyOTF):
    def __init__(self, dataset, idxs_lb, net, handler, args):
        super(RandomSamplingOTF, self).__init__(dataset, idxs_lb, net, handler, args)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
