import shelve
from features.feature import Feature


class FeatureWithStorage(Feature):

    def __init__(self, base_feature, store_file):
        self.store_file = store_file
        self.base_feature = base_feature
        self.base_feature_name = type(base_feature).__name__
        self.extract_history = self.load_extract_history()

    def load_extract_history(self):
        extract_history = shelve.open(self.store_file)
        return extract_history

    def __del__(self):
        self.extract_history.close()

    def extract_feature(self, user_id, user_tweets, force_new_download=False):
        key = str((user_id, self.base_feature_name))
        # If entry does not exists
        if (key not in self.extract_history) or force_new_download:
            self.extract_history[key] = self.base_feature.extract_feature(user_id, user_tweets)

        return self.extract_history[key]
