from validation.k_fold_base import KFoldBase
from sklearn.model_selection import GridSearchCV


class NestedKFoldValidation(KFoldBase):
    def __init__(self, param_grid, k=5, k2=3, progress_bar=None, random_state=None):
        super().__init__(k, progress_bar, random_state)
        self.k2 = k2
        self.param_grid = param_grid

    def predict_abstract(self, clf_pipeline, X_train, y_train, X_test):
        gs = GridSearchCV(clf_pipeline, self.param_grid, cv=self.k2, n_jobs=1)
        gs.fit(X_train, y_train)

        # print gs.best_estimator_
        return gs.predict(X_test)
