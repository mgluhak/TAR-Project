from evaluation.eval_utils import store_intermediate_step
from evaluation.eval_utils import load_intermediate_step
from reduction.reduction_base import Reduction
import os


class ReductionWithStorage(Reduction):
    def function_abstract(self, X, y):
        return self.base_reduction.function_abstract(X, y)

    def __init__(self, base_reduction, base_clf, threshold, count_nan=True, progress_bar=None, load_stored=True, store_new=True):
        super().__init__(threshold, count_nan, progress_bar)
        self.base_reduction = base_reduction
        self.name = "{0}-{1}".format(str(type(base_reduction).__name__), str(type(base_clf).__name__))
        self.load_stored = load_stored
        self.store_new = store_new

    def reduce(self, features, y, names):
        if self.load_stored and os.path.exists(os.getcwd() + '/cache/' + self.name + '_features.pkl'):
            features = load_intermediate_step('cache/' + self.name + '_features.pkl')
        else:
            features = super().reduce(features, y, names)
            if self.store_new:
                store_intermediate_step(features, 'cache/' + self.name + '_features.pkl')

        return features
