from validation.k_fold_base import KFoldBase


class KFoldValidation(KFoldBase):
    def __init__(self, k=5, progress_bar=None, random_state=None):
        super().__init__(k, progress_bar, random_state)

    def predict_abstract(self, clf_pipeline, X_train, y_train, X_test):
        clf_pipeline.fit(X_train, y_train)

        return clf_pipeline.predict(X_test)
