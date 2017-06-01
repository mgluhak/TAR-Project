from features.feature import Feature
import dataset.dataset_reader as dr
import enchant


class WordRichnessFeature(Feature):
    def __init__(self):
        self.d = enchant.Dict("en_US")

    @staticmethod
    def calculate_uniqueness(word_map):
        unique = 0
        total = 0
        for key in word_map:
            if word_map[key] == 1:
                unique += 1
            total += 1
        return unique / total

    def extract_feature(self, user_id, user_tweets):
        word_map = {}
        for tweet in user_tweets:
            for word in tweet:
                if self.d.check(word):
                    if word in word_map:
                        word_map[word] += 1
                    else:
                        word_map[word] = 1
        return self.calculate_uniqueness(word_map)

#data = dr.load_dataset()
#wrf = WordRichnessFeature()
#print(wrf.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
