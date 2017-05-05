from Feature import Feature

import pickle
import sys
sys.path.insert(1, '../dataset')
import dataset_reader as dr

class CapSentenceFeature(Feature):



    def extract_feature(self, user, tweets):
        tweet_count = len(tweets)
        
        if tweet_count == 0:
            self.map[user] = 0
            return

        capital_count = 0.0
        for tweet in tweets:
            for word in tweet:
                #skip tweet handles punctuation and hashtags
                if(word.isalpha()):
                    if word[0].isupper():
                        capital_count+=1
                    break

        self.map[user] = capital_count/tweet_count


data = dr.load_dataset()
CSF = CapSentenceFeature()
for user in data:
    CSF.extract_feature(user, data[user].get_tweets())

print(CSF.retrieve('36b2593435e1bed13eb138c1973c13ed'))
