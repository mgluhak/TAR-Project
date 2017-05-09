from features.feature import Feature
from features.feature_storage import FeatureWithStorage

import dataset.dataset_reader as dr


class EndsWithInterpunctionFeature(Feature):

    def extract_feature(self, user, tweets):

        tweet_count = len(tweets)
        if tweet_count == 0:
            #self.map[user] = 0
            return 0

        interpuncted_tweets = 0.0
        for tweet in tweets:
            if len(tweet) == 0:
                continue
            final_token = tweet[-1]
            if final_token == "?" or final_token == "." or final_token == "!":
                interpuncted_tweets += 1.0

        #self.map[user] = interpuncted_tweets / tweet_count
        return interpuncted_tweets / tweet_count


#data = dr.load_dataset()
#EWF = FeatureWithStorage(EndsWithInterpunctionFeature(), 'abc.shelve')
#print(EWF.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
#for user in data:
#    EWF.extract_feature(user, data[user].get_tweets())

#a1_sorted_keys = sorted(EWF.map, key=EWF.map.get, reverse=True)
#for r in a1_sorted_keys:
#    print(r, EWF.map[r])
