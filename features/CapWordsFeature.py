from Feature import Feature

import pickle
import sys
sys.path.insert(1, '../dataset')
import dataset_reader as dr

class CapWordsFeature(Feature):



    def extract_feature(self, user, tweets):
        tweet_count = len(tweets)
        
        if tweet_count == 0:
            self.map[user] = 0
            return

        tweet_capitalisation = []
        for tweet in tweets:
            capital_count = 0.0
            for word in tweet:
                if(word.isalpha()):
                    if word[0].isupper():
                        capital_count+=1
            length = len(tweet)
            if (length!=0):
                tweet_capitalisation.append(capital_count/len(tweet))    

        self.map[user] = sum(tweet_capitalisation)/len(tweet_capitalisation)


data = dr.load_dataset()
CWF = CapWordsFeature()
for user in data:
    CWF.extract_feature(user, data[user].get_tweets())

print(CWF.retrieve('36b2593435e1bed13eb138c1973c13ed'))

