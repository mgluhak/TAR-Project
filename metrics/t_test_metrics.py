from metrics.metrics_base import BaseMetrics
from pandas import DataFrame
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel

from metrics.metrics_map import MetricsMap
from metrics.standard_metrics import StandardMetrics


class TTestMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.standard = StandardMetrics()

    def show_result(self, metrics_map):
        value_map_1 = metrics_map[0].retrieve()
        value_map_2 = metrics_map[1].retrieve()

        for key in value_map_1:
            self.result_map[key] = self.perform_metrics_abstract((value_map_1[key], value_map_2[key]))

        return self.show_result_abstract()

    def show_result_abstract(self):
        return self.convert_to_pandas(self.result_map).head()

    def perform_metrics_abstract(self, value_map_combined):
        value_map_1 = value_map_combined[0]
        value_map_2 = value_map_combined[1]

        accuracy_1, precision_1, recall_1, f1_1 = self.standard.perform_metrics_abstract(value_map_1)
        accuracy_2, precision_2, recall_2, f1_2 = self.standard.perform_metrics_abstract(value_map_2)

        return ttest_rel(f1_1, f1_2)

    @staticmethod
    def convert_to_pandas(result_map):
        df = DataFrame()
        index_list = []
        for key in result_map:
            df = df.append([result_map[key]])
            index_list.append(key)
        df.index = index_list
        df.columns = ['Statistics', 'p-value']
        return df

mm = MetricsMap('/home/mihael/Documents/8. semestar/APT/Projekt/bas pravi git/TAR-Project/evaluation/svm_linear_features.pkl')
mm2 = MetricsMap('/home/mihael/Documents/8. semestar/APT/Projekt/bas pravi '
                 'git/TAR-Project/evaluation/logistic_regression.pkl')

tt = TTestMetrics()
tt.show_result(metrics_map=(mm, mm2))