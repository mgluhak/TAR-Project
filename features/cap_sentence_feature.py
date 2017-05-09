from features.feature import Feature

import dataset.dataset_reader as dr


class CapSentenceFeature(Feature):
    def extract_feature(self, user, tweets):
        tweet_count = len(tweets)

        if tweet_count == 0:
            #self.map[user] = 0
            return 0

        capital_count = 0.0
        for tweet in tweets:
            for word in tweet:
                # skip tweet handles punctuation and hashtags
                if word.isalpha() and word != "NUMBER" and word != "URL":
                    if word[0].isupper():
                        capital_count += 1
                    break

        #self.map[user] = capital_count / tweet_count
        return capital_count / tweet_count

#data = dr.load_dataset()
#CSF = CapSentenceFeature()
#print(CSF.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
#for user in data:
#    CSF.extract_feature(user, data[user].get_tweets())

#print(CSF.retrieve('36b2593435e1bed13eb138c1973c13ed'))
#print(data['36b2593435e1bed13eb138c1973c13ed'].get_tweets())
