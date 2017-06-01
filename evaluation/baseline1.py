import dataset.dataset_reader as dr
from evaluation.eval_utils import nested_k_fold_cv
from evaluation.eval_utils import old_store_result
from evaluation.eval_utils import get_documents_y
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from features.cap_letters_feature import CapLettersFeature
from features.cap_sentence_feature import CapSentenceFeature
from features.cap_words_feature import CapWordsFeature
from features.ends_with_interpunction_feature import EndsWithInterpunctionFeature
from features.punctuation_count_feature import PunctuationCountFeature
from features.utility import penn_to_wn
from evaluation.eval_utils import get_all_features

# custom spliter used instead of a tokenizer, since the tweets are already tokenized
def spaceSplitter(list):
    return list.split(" ")


def getFeatures(dataset, classification="both"):
    # 1. ucitavanje dataseta
    documents, y = get_documents_y(dataset, classification)

    ## Definining tf-idf vector
    vectorizer = TfidfVectorizer(tokenizer=spaceSplitter)
    vectorizer.fit(documents)

    features = vectorizer.transform(documents)
    return features, y


# clasification - possible modes - age, gender, both
def evaluate(classification="both"):
    dataset = dr.load_dataset()
    features, y = getFeatures(dataset, classification)

    print(features.shape)

    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)

    yLabel = encoder.transform(y)

    ## Training linear SVM

    pot2func = lambda x: 2 ** x
    pot2 = map(pot2func, range(-5, 5))
    param_grid = {'svc__C': list(pot2)}
    clfSVM = LinearSVC()
    pipeline = Pipeline([('svc', clfSVM)])

    ## K- fold validation
    return nested_k_fold_cv(pipeline, param_grid, features, yLabel, k1=5, k2=3)


    # evaluation results

    # age only
    # accuracy,precisionMacro,recallMacro,f1Macro
    # 0.451974730696 0.217529421246 0.256729323308 0.216938342466

    # gender only
    # accuracy,precisionMacro,recallMacro,f1Macro
    # 0.717965367965 0.720882394348 0.717965367965 0.717075870313

    # age & gender
    # accuracy,precisionMacro,recallMacro,f1Macro
    # 0.318092414832 0.237126847568 0.22628968254 0.213770186078

    # store_result((evaluate("gender")), 'results/baseline3_9_5_2017.pkl', "Gender only")
    # store_result((evaluate("age")), 'results/baseline3_9_5_2017.pkl', "Age only")
    # store_result((evaluate("both")), 'results/baseline3_9_5_2017.pkl', "Both")

# proba
# additional = [CapSentenceFeature(), CapLettersFeature(), CapWordsFeature(), EndsWithInterpunctionFeature(),
#                 PunctuationCountFeature()]
# print(get_all_features(additional, dr.load_dataset()))