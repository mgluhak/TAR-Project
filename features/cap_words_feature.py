import sys

from features.feature import Feature
import dataset.dataset_reader as dr


class CapWordsFeature(Feature):
    def extract_feature(self, user, tweets):
        tweet_count = len(tweets)

        if tweet_count == 0:
            #self.map[user] = 0
            return 0

        tweet_capitalisation = []
        for tweet in tweets:
            capital_count = 0.0
            for word in tweet:
                if word.isalpha() and word != "NUMBER" and word != "URL":
                    if word[0].isupper():
                        capital_count += 1
            length = len(tweet)
            if length != 0:
                tweet_capitalisation.append(capital_count / len(tweet))

        #self.map[user] = sum(tweet_capitalisation) / len(tweet_capitalisation)
        return sum(tweet_capitalisation) / len(tweet_capitalisation)

#data = dr.load_dataset()
#CWF = CapWordsFeature()
#print(CWF.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
#for user in data:
#    CWF.extract_feature(user, data[user].get_tweets())

#print(CWF.map)
