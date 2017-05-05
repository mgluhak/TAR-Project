from features.feature import Feature
import dataset.dataset_reader as dr


class AverageSentenceLengthFeature(Feature):

    def extract_feature(self, user, tweets):
        
        tweet_count = len(tweets)
        if tweet_count == 0:
            self.map[user] = 0
            return

        sentence_lengths = 0.0
        for tweet in tweets:
            sentence_lengths += len(tweet)

        self.map[user] = sentence_lengths/tweet_count


data = dr.load_dataset()
ASLF = AverageSentenceLengthFeature()
for user in data:
    ASLF.extract_feature(user, data[user].get_tweets())

a1_sorted_keys = sorted(ASLF.map, key=ASLF.map.get, reverse=True)
for r in a1_sorted_keys:
    print(r, ASLF.map[r])

print(data['4392ce5936db3591f713b0171ffd0e17'].get_age_group())
