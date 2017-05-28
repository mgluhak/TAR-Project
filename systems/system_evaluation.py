import abc
from evaluation.eval_utils import *
from sklearn.pipeline import Pipeline


class EvaluationSystem:
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def get_clf(self, classification):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractclassmethod
    def get_features(self, dataset, classification):
        raise NotImplementedError("Please Implement this method")

    def evaluate(self, dataset, validation, classification="both", additional_features=None, reduction=None, scl=None, pca=None):
        X, y, names = self.get_features(dataset, classification)

        if additional_features is not None and len(additional_features) > 0:
            other_features = get_all_features(additional_features, dataset)
            X = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(other_features), X]))

        if reduction is not None:
            X = reduction.reduce(features=X, y=y, names=names)

        encoder = preprocessing.LabelEncoder()
        encoder.fit(y)
        y_label = encoder.transform(y)

        # classificator
        #clf_pipeline = self.get_clf(classification)
        pipe_list = []
        if scl is not None:
            pipe_list.append(('scl', scl))
        if pca is not None:
            pipe_list.append(('pca', pca))
            X = X.todense()
        pipe_list.append(self.get_clf(classification))

        pipe_clf = Pipeline(pipe_list)
        return validation.validate(pipe_clf, X, y_label)

        #if param_grid is not None and len(param_grid) > 0:
        #    return nested_k_fold_cv(clf_pipeline, param_grid, X, y_label, k1, k2)
        #else:
        #    return k_fold(clf_pipeline, X, y_label, k1)
