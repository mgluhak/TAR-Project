from features.feature import Feature

import dataset.dataset_reader as dr


class CapLettersFeature(Feature):
    def extract_feature(self, user, tweets):
        tweet_count = len(tweets)

        if tweet_count == 0:
            return 0

        tweet_list = []
        for tweet in tweets:
            upper_count = 0.0
            lower_count = 0.0
            for word in tweet:
                # skip tweet handles punctuation and hashtags
                if word.isalpha() and word != "NUMBER" and word != "URL":
                    for char in word:
                        if char.isupper():
                            upper_count += 1
                        else:
                            lower_count += 1

            if upper_count == 0.0 or (upper_count + lower_count) == 0.0:
                tweet_list.append(0)
            else:
                tweet_list.append(upper_count / (upper_count + lower_count))

        #self.map[user] = sum(tweet_list) / (len(tweet_list))
        return sum(tweet_list) / (len(tweet_list))


#data = dr.load_dataset()
#CLF = CapLettersFeature()
#print(CLF.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
#for user in data:
#    CLF.extract_feature(user, data[user].get_tweets())

#a1_sorted_keys = sorted(CLF.map, key=CLF.map.get, reverse=True)
#for r in a1_sorted_keys:
#    print(r, CLF.map[r])
