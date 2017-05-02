import os
import pickle


class OutOfDictWordsFeature:
    def __init__(self):
        self.map = None
        self.dir = os.path.dirname(__file__)
        self.load(self)

    @staticmethod
    def load(self):
        input_file = open(os.path.join(self.dir, './pkls/out_of_dict_map.pkl'), 'rb')
        self.map = pickle.load(input_file)
        input_file.close()

    def retrieve(self):
        return self.map


# tweets = dr.load_dataset()
# odwf = OutOfDictWordsFeature()
# for user in tweets:
#     print(odwf.retrieve()[user])
#     break
