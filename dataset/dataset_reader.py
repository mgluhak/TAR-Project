import os
import pickle
import sys
import dataset.dataset_map_entry as me


def load_dataset():
    input_dataset = open(os.path.join(os.path.dirname(__file__), './output/map_final.pkl'), 'rb')
    sys.modules['dataset_map_entry'] = me
    dataset = pickle.load(input_dataset)
    input_dataset.close()

    return dataset
