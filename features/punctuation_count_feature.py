from features.feature import Feature
from features.feature_storage import FeatureWithStorage

import numpy as np
import dataset.dataset_reader as dr


class PunctuationCountFeature(Feature):
    def __init__(self):
        super().__init__()
        #       self.map = {}
        self.punctuation_mapper = {
            ",": 0,
            ".": 1,
            ";": 2,
            "!": 3,
            "?": 4,
            "-": 5
        }
        self.punc_marks = [',', '.', ';', '!', '?', '-']

    def extract_feature(self, user, tweets):

        tweet_count = len(tweets)
        if tweet_count == 0:
            # self.map[user] = list(zip(self.punc_marks, np.zeros(6)))
            return np.zeros(6)
            #return dict(zip(self.punc_marks, np.zeros(6)))

        total_counts = np.zeros(6)
        for tweet in tweets:
            counts = np.zeros(6)
            char_count = 0
            for word in tweet:
                if word != "NUMBER" or word != "URL":
                    char_count += len(word)
                    for char in word:
                        if char in self.punc_marks:
                            counts[self.punctuation_mapper[char]] += 1.0
            if char_count == 0:
                continue
            counts /= char_count
            total_counts += counts

        total_counts /= tweet_count
        # self.map[user] = dict(zip(self.punc_marks, total_counts))

        #return dict(zip(self.punc_marks, total_counts))
        return total_counts


#data = dr.load_dataset()
#PCF = FeatureWithStorage(PunctuationCountFeature(), 'abc.shelve')
#print(PCF.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
#for user in data:
#    PCF.extract_feature(user, data[user].get_tweets())
#    break

# for user in PCF.map.keys():
#    print(user, PCF.map[user])
