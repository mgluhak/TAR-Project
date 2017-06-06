import sys

from features.feature import Feature
import dataset.dataset_reader as dr
import pickle
import numpy as np


class GloveFeatures(Feature):
    def __init__(self, path="cache/w2v_twitter100.pkl"):
        self.w2v_dict = pickle.load(open(path, "rb"))

    def extract_feature(self, user, tweets):
        feature = np.zeros(100)
        for tweet in tweets:
            for word in tweet:
                if word in self.w2v_dict:
                    feature += self.w2v_dict[word]
            length = len(tweet)
        return feature / len(tweets) #if len(tweets) > 0 else np.ones(100)
