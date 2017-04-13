import pickle
from nltk.tag import pos_tag

class POS_tagger:
    def __init__(self):
        self.pos_tagger = pickle.load( open("pos_tagger.pkl","r") )

    def tag(self,tokens,tagger="TnT"):
        if tagger == "TnT":
            return self.pos_tagger.tag(tokens)
        elif tagger == "Perceptron":
            return pos_tag(tokens)
        else:
            raise ValueError("Tagger " + tagger + " is not supported")