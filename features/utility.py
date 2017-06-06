import pickle
from nltk.tag import pos_tag
import nltk
from nltk.corpus import wordnet as wn


class POS_tagger:
    def __init__(self):
        self.pos_tagger = pickle.load(open("cache/pos_tagger.pkl", "rb"))

    def tag(self, tokens, tagger="TnT"):
        if tagger == "TnT":
            return self.pos_tagger.tag(tokens)
        elif tagger == "Perceptron":
            return pos_tag(tokens)
        else:
            raise ValueError("Tagger " + tagger + " is not supported")


# taken from stackoverflow
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN
