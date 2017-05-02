import os
import pickle
import enchant
import dataset.dataset_reader as dr
from Levenshtein.StringMatcher import StringMatcher

# Levensthein word distance ratio threshold
WORD_DIST_RATIO = 0.6


class OutOfDictWordsFeature:
    def __init__(self):
        self.map = None
        self.brands = None
        self.dir = os.path.dirname(__file__)
        self.load_files()
        # dictionary with english words
        self.d = enchant.Dict("en_US")
        # dictionary with spanish words (because they may occurr in english tweets)
        self.d_sp = enchant.Dict("es")
        # Tool for calculating Levenshtein word distance
        self.sm = StringMatcher()

    def load_files(self):
        input_file = open(os.path.join(self.dir, './pkls/out_of_dict_map.pkl'), 'rb')
        self.map = pickle.load(input_file)
        input_file.close()
        # load brands
        self.brands = set(line.lower().strip() for line in open(os.path.join(self.dir, './extraction/brands.txt')))

    def extract_out_of_dict_word_ratio(self, user, tweets):
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
        self.map[user] = (len(out_of_dict_words), len(out_of_dict_words) + len(in_dict_words))

    def retrieve(self, user_id, tweets=None):
        if (user_id in self.map) and tweets is not None:
            self.extract_out_of_dict_word_ratio(user_id, tweets)
            return self.map[user_id]
        elif user_id in self.map:
            return self.map[user_id]
        else:
            raise ValueError("User with given id does not exist!")

tweets = dr.load_dataset()
odwf = OutOfDictWordsFeature()
for user in tweets:
    print(user)
    print(odwf.retrieve('36b2593435e1bed13eb138c1973c13ed'))
    break
