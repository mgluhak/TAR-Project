import pickle
from demonstration.eval_utils import get_documents_y


def spaceSplitter(list):
    return list.split(" ")



def get_features(dataset, classification="both"):
    documents, y = get_documents_y(dataset, classification)
    features = vectorizer.transform(documents)
    return features, y

def get_features_for(fname):
    tweets = pickle.load(open(fname, "rb"))

    documents = []
    document = []
    for tweet in tweets:
        document.append(" ".join(tweet))
    documents.append(" ".join(document))

    X = vectorizer.transform(documents)

    return X

vectorizer = pickle.load(open("tf-idf_vectorizer.pkl", "rb"))
