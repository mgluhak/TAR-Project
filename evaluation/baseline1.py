import dataset.dataset_reader as dr
from evaluation.eval_utils import nested_k_fold_cv
from evaluation.eval_utils import store_result
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np

# custom spliter used instead of a tokenizer, since the tweets are already tokenized
def spaceSplitter(list):
    return list.split(" ")


def getFeatures(dataset,classification="both"):

    documents = []
    y = []

    # joining tokens with whitespace in order to fit the tf-idf vectorizer
    for user in dataset:
        tweets = dataset[user].get_tweets()
        if classification == "both":
            y.append(str(dataset[user].get_gender().value) + str(dataset[user].get_age_group().value))
        elif classification == "gender":
            y.append(str(dataset[user].get_gender().value))
        elif classification == "age":
            y.append(str(dataset[user].get_age_group().value))
        else:
            raise ValueError("Given clasification taks is not specified")

        document = []
        for tweet in tweets:
            document.append(" ".join(tweet))
        documents.append(" ".join(document))

    ## Definining tf-idf vector

    vectorizer = TfidfVectorizer(tokenizer=spaceSplitter)
    vectorizer.fit(documents)

    features = vectorizer.transform(documents)
    return features,y

# clasification - possible modes - age, gender, both
def evaluate(classification="both"):
    dataset = dr.load_dataset()
    features,y = getFeatures(dataset,classification)

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

store_result((evaluate("gender")), 'results/baseline_9_5_2017.pkl', "Gender only")
store_result((evaluate("age")), 'results/baseline_9_5_2017.pkl', "Age only")
store_result((evaluate("both")), 'results/baseline_9_5_2017.pkl', "Both")