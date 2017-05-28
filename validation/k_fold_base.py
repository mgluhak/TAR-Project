import abc
from validation.validation_base import Validation
from sklearn.model_selection import StratifiedKFold


class KFoldBase(Validation):
    __metaclass__ = abc.ABCMeta

    def __init__(self, k=5, progress_bar=None, random_state=None):
        super().__init__(progress_bar, random_state)
        self.k = k

    def validate(self, clf_pipeline, X, y):
        kf1 = StratifiedKFold(n_splits=self.k, random_state=self.random_state)
        y_trues = []
        y_predicted = []

        self.progress_bar.initialize(self.k)
        for train_index, test_index in kf1.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.progress_bar.increase_counter()

            y_trues.append(y_test)
            y_predicted.append(self.predict_abstract(clf_pipeline, X_train, y_train, X_test))

        return y_trues, y_predicted

    @abc.abstractclassmethod
    def predict_abstract(self, clf_pipeline, X_train, y_train, X_test):
        raise NotImplementedError("Please Implement this method")

