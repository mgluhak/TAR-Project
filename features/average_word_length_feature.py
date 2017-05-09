from features.feature import Feature
import dataset.dataset_reader as dr


class AverageWordLengthFeature(Feature):
    def extract_feature(self, user, tweets):

        tweet_count = len(tweets)
        if tweet_count == 0:
            #self.map[user] = 0
            return 0

        word_lengths = 0.0
        word_count = 0
        for tweet in tweets:
            for word in tweet:
                if word.isalpha() and word != "NUMBER" and word != "URL":
                    word_lengths += len(word)
                    word_count += 1

        if word_count == 0:
            #self.map[user] = 0
            return 0

        #self.map[user] = word_lengths / word_count
        return word_lengths / word_count


#data = dr.load_dataset()
#AWLF = AverageWordLengthFeature()
#print(AWLF.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
#for user in data:
#    AWLF.extract_feature(user, data[user].get_tweets())

#a1_sorted_keys = sorted(AWLF.map, key=AWLF.map.get, reverse=True)
#for r in a1_sorted_keys:
#    print(r, AWLF.map[r])

#print(data['d912c51b671d209edc6672fe203dd345'].get_age_group())
