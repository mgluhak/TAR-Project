import abc


class Feature:
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def extract_feature(self, user_id, user_tweets):
        pass
