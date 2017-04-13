from nltk.corpus import treebank
from nltk.tag import tnt
import pickle

#treniranje TnT part of speech taggera

train_set = treebank.tagged_sents()[:3000]
test_set = treebank.tagged_sents()[3000:]

pos_tagger = tnt.TnT()
pos_tagger.train(train_set)

output_file = open("pos_tagger.pkl", "w")
pickle.dump(pos_tagger, output_file)
output_file.close()