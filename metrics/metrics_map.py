from evaluation.eval_utils import store_intermediate_step
from evaluation.eval_utils import load_intermediate_step


class MetricsMap:
    def __init__(self, path=None):
        if path is not None:
            self.result_map = load_intermediate_step(path)
        else:
            self.result_map = {}

    def evaluate(self, dataset, system, validation, classification, additional_features=None, reduction=None, scl=None,
                 pca=None):
        y_trues, y_predicted = system.evaluate(dataset, validation, classification, additional_features, reduction, scl,
                                               pca)
        self.result_map[classification] = (y_trues, y_predicted)

    def retrieve(self):
        return self.result_map

    def save_map(self, path):
        store_intermediate_step(self.result_map, path)
