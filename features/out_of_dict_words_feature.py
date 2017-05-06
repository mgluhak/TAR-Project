import os
import enchant
import dataset.dataset_reader as dr
from features.feature import Feature
from features.feature_storage import FeatureWithStorage
from Levenshtein.StringMatcher import StringMatcher

# Levensthein word distance ratio threshold
WORD_DIST_RATIO = 0.6
BRANDS_FILE = './additional/brands.txt'


class OutOfDictWordsFeature(Feature):
    def __init__(self):
        self.brands = None
        self.dir = os.path.dirname(__file__)
        # dictionary with english words
        self.d = enchant.Dict("en_US")
        # dictionary with spanish words (because they may occur in english tweets)
        self.d_sp = enchant.Dict("es")
        # Tool for calculating Levenshtein word distance
        self.sm = StringMatcher()
        # load brands
        self.brands = set(line.lower().strip() for line in open(os.path.join(self.dir, BRANDS_FILE)))

    def extract_feature(self, user_id, tweets):
        # List with out of dictionary words
        out_of_dict_words = set()
        # List with words in dictionary
        in_dict_words = set()
        for tweet in tweets:
            for word in tweet:
                if word.replace('\'', '').replace('-', '').isalpha() and not self.d.check(
                        word) and not self.d_sp.check(word) and (word not in self.brands) and ('URL' not in word) and (
                            'NUMBER' not in word):
                    self.sm.set_seq1(seq1=word)
                    founded = False
                    for suggestion in self.d.suggest(word):
                        self.sm.set_seq2(seq2=suggestion)
                        if self.sm.ratio() > WORD_DIST_RATIO:
                            out_of_dict_words.add(word)
                            founded = True
                            break
                    if not founded:
                        in_dict_words.add(word)
                else:
                    in_dict_words.add(word)
        return len(out_of_dict_words), len(out_of_dict_words) + len(in_dict_words)


tweets = dr.load_dataset()
o1 = OutOfDictWordsFeature()
print(type(o1).__name__)
odwf = FeatureWithStorage(OutOfDictWordsFeature(), 'abc.shelve')
for user in tweets:
    print(user)
    print(odwf.extract_feature('36b2593435e1bed13eb138c1973c13ed', tweets['36b2593435e1bed13eb138c1973c13ed'].tweets))
    break
