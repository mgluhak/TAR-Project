import abc
from progress.pyprind_progress import PyprindProgressBar


class Validation:
    __metaclass__ = abc.ABCMeta

    def __init__(self, progress_bar=None, random_state=None):
        self.progress_bar = progress_bar if progress_bar is not None else PyprindProgressBar()
        self.random_state = random_state

    @abc.abstractclassmethod
    def validate(self, clf_pipeline, X, y):
        raise NotImplementedError("Please Implement this method")
