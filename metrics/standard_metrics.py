from metrics.metrics_base import BaseMetrics
import numpy as np
from pandas import DataFrame
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


class StandardMetrics(BaseMetrics):
    def __init__(self, average="macro"):
        super().__init__()
        self.average = average

    def show_result_abstract(self):
        return self.convert_to_pandas(self.average_results(self.result_map)).head()

    def perform_metrics_abstract(self, value_map):
        accuracy = []
        precision = []
        recall = []
        f1 = []
        y_trues = value_map[0]
        y_predicted = value_map[1]

        for y_true, y_pred in zip(y_trues, y_predicted):
            accuracy.append(accuracy_score(y_true, y_pred))
            precision.append(precision_score(y_true, y_pred, average=self.average))
            recall.append(recall_score(y_true, y_pred, average=self.average))
            f1.append(f1_score(y_true, y_pred, average=self.average))

        return accuracy, precision, recall, f1

    @staticmethod
    def convert_to_pandas(result_map):
        df = DataFrame()
        index_list = []
        for key in result_map:
            df = df.append([result_map[key]])
            index_list.append(key)
        df.index = index_list
        df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        return df

    @staticmethod
    def average_results(result_map):
        avg_res_map = {}
        for key in result_map:
            avg_res_map[key] = (np.average(res) for res in result_map[key])
        return avg_res_map
