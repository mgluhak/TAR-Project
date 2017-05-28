import abc
import numpy as np


class BaseMetrics:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.result_map = {}

    def show_result(self, metrics_map):
        value_map = metrics_map.retrieve()
        for key in value_map:
            self.result_map[key] = self.perform_metrics_abstract(value_map[key])
        return self.show_result_abstract()

    def show_last_result(self):
        return self.show_result_abstract()

    @abc.abstractclassmethod
    def perform_metrics_abstract(self, value_map):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractclassmethod
    def show_result_abstract(self):
        raise NotImplementedError("Please Implement this method")
