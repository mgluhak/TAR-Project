import abc
import numpy as np
from progress.silent_progress import SilentProgressBar


class Reduction:
    __metaclass__ = abc.ABCMeta

    def __init__(self, threshold, count_nan=True, progress_bar=None):
        self.threshold = threshold
        self.count_nan = count_nan
        self.progress_bar = progress_bar if progress_bar is not None else SilentProgressBar()

    def reduce(self, features, y, names):
        # new_names = []
        # DEBUG
        f = features.tolil()
        total = len(names)
        self.progress_bar.initialize(total)

        for ig, name in zip(self.function_abstract(features.todense(), y), names):
            if (ig > self.threshold) or (self.count_nan and np.isnan(ig)):
                # new_names.append(name)
                continue
            else:
                column = names.index(name)
                # uklanjanje
                f[:, column] = 0
            self.progress_bar.increase_counter()
        return f.todense()

    @abc.abstractclassmethod
    def function_abstract(self, X, y):
        raise NotImplementedError("Please Implement this method")
