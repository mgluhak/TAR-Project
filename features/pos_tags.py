from features.feature import Feature
from features.utility import POS_tagger
import dataset.dataset_reader as dr
from collections import Counter
import numpy as np


class PosTagFeature(Feature):
    def __init__(self, type="Perceptron"):
        self.tagger = POS_tagger()
        self.type = type

    def extract_feature(self, user, tweets):
        tweet_count = len(tweets)

        if self.type == "Perceptron":
            possibleTags = ['WP$', 'VBN', 'WRB', 'TO', 'VBP', 'POS', 'NNPS', 'PRP', 'UH', 'RBS', 'CD', 'DT', 'RB', 'IN',
                            'JJ', 'NNP', 'WDT', 'CC', 'RP', 'VBG', '``', "''", 'VB', '$', '.', 'PDT', 'JJS', 'VBD',
                            'JJR', 'FW', ')', 'WP', ':', 'LS', '#', 'SYM', '(', 'RBR', 'MD', 'VBZ', 'NNS', 'NN', 'PRP$',
                            ',', 'EX']
            counter = Counter(possibleTags)
        else:
            counter = Counter()

        if tweet_count == 0:
            return list(map(lambda x: x[1], counter.items()))

        token_count = 0

        for tweet in tweets:
            tagged_tweet = self.tagger.tag(tweet, tagger=self.type)
            for (word, tag) in tagged_tweet:
                counter[tag] += 1
                # if tag in dict:
                #    dict[tag] += 1
                # else:
                #    dict[tag] = 1
            token_count += len(tweet)

        for tag in counter:
            counter[tag] /= token_count

        #return list(map(lambda x: x[1], counter.items()))
        return np.array([x[1] for x in counter.items()])

        # data = dr.load_dataset()
        # #
        # from features.feature_storage import FeatureWithStorage
        # PosFeatures = FeatureWithStorage( PosTagFeature(type="Perceptron"),'pos2.shelve')
        #
        # for user in data:
        #     counter = PosFeatures.extract_feature(user, data[user].get_tweets())
        #     print (counter)
        #     #allCounter = allCounter + counter

        # print (list( map(lambda x: x[1],counter.items())))
        # print (allCounter.keys())
        # print (allCounter)
