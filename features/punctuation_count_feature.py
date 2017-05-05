from features.feature import Feature
import dataset.dataset_reader as dr
import numpy as np


class PunctuationCountFeature(Feature):

    def __init__(self):
        self.map = {}
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
            self.map[user] = list(zip(self.punc_marks, np.zeros(6)))
            return

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
        self.map[user] = dict(zip(self.punc_marks, total_counts))


data = dr.load_dataset()
PCF = PunctuationCountFeature()
for user in data:
    PCF.extract_feature(user, data[user].get_tweets())

for user in PCF.map.keys():
    print(user, PCF.map[user])
