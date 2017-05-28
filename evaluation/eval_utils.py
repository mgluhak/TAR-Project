import os
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle
import nltk
import dataset.dataset_reader as dr
from features.utility import penn_to_wn
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn import preprocessing
import numpy as np
from scipy import sparse


def print_evaluation_results(pipeline, parameters, X, y, k1=10, k2=3):
    accuracy, precision, recall, f1 = nested_k_fold_cv(pipeline, parameters, X, y, k1=k1, k2=k2)
    # print(accuracy)
    # print(precision)
    # print(recall)
    # print(f1)
    print("accuracy,precisionMacro,recallMacro,f1Macro")
    print(np.average(accuracy), np.average(precision), np.average(recall), np.average(f1))


def get_documents_y(dataset, classification="both", processing=None):
    y = []
    documents = []
    for user in sorted(dataset.keys()):
        tweets = dataset[user].get_tweets()
        if classification == "both":
            y.append(str(dataset[user].get_gender().value) + str(dataset[user].get_age_group().value))
        elif classification == "gender":
            y.append(str(dataset[user].get_gender().value))
        elif classification == "age":
            y.append(str(dataset[user].get_age_group().value))
        else:
            raise ValueError("Given clasification taks is not specified")

        if processing == "stemming":
            stemmer = SnowballStemmer('english')
        elif processing == "lemmatization":
            lemmatizer = WordNetLemmatizer()

        document = []
        for tweet in tweets:
            if processing == "stemming":
                tweet = map(lambda x: stemmer.stem(x), tweet)
            elif processing == "lemmatization":
                taggedTweet = nltk.pos_tag(tweet)
                tweet = map(lambda x: lemmatizer.lemmatize(x[0], penn_to_wn(x[1])), taggedTweet)
            document.append(" ".join(tweet))
        documents.append(" ".join(document))

    return documents, y


# def evaluate(dataset, clf_pipeline, classification="both",
#              additional_features=None, param_grid=None, k1=5, k2=3):
#     X, y = None, None
#
#     if additional_features is not None and len(additional_features) > 0:
#         other_features = get_all_features(additional_features, dataset)
#         X = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(other_features), X]))
#
#     encoder = preprocessing.LabelEncoder()
#     encoder.fit(y)
#     y_label = encoder.vectorizer(y)
#
#     if param_grid is not None and len(param_grid) > 0:
#         return nested_k_fold_cv(clf_pipeline, param_grid, X, y_label, k1, k2)
#     else:
#         return k_fold(clf_pipeline, X, y_label, k1)


def k_fold(clf, X, y, k=10):
    kf1 = StratifiedKFold(n_splits=k, random_state=42)

    accuracy = []
    precision = []
    recall = []
    f1 = []

    counter = 1  # DEBUG
    for train_index, test_index in kf1.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("iteration = " + str(counter) + "/" + str(k))  # DEBUG
        counter += 1

        clf.fit(X_train, y_train)

        # print gs.best_estimator_
        y_predicted = clf.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_predicted))
        precision.append(precision_score(y_test, y_predicted, average="macro"))
        recall.append(recall_score(y_test, y_predicted, average="macro"))
        f1.append(f1_score(y_test, y_predicted, average="macro"))

    return accuracy, precision, recall, f1


def nested_k_fold_cv(clf, param_grid, X, y, k1=10, k2=3):
    # kf1 = KFold(n_splits=k1)
    kf1 = StratifiedKFold(n_splits=k1, random_state=42)

    accuracy = []
    precision = []
    recall = []
    f1 = []

    counter = 1  # DEBUG
    for train_index, test_index in kf1.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("iteration = " + str(counter) + "/" + str(k1))  # DEBUG
        counter += 1

        gs = GridSearchCV(clf, param_grid, cv=k2, n_jobs=1)
        gs.fit(X_train, y_train)

        # print gs.best_estimator_
        y_predicted = gs.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_predicted))
        precision.append(precision_score(y_test, y_predicted, average="macro"))
        recall.append(recall_score(y_test, y_predicted, average="macro"))
        f1.append(f1_score(y_test, y_predicted, average="macro"))

    return accuracy, precision, recall, f1


def old_load_saved_result(path):
    input_hist = open(path, 'rb')
    results = pickle.load(input_hist)
    input_hist.close()
    # { 'label' : (acc, precision, recall, f1) }
    return results


def old_store_result(results, store_file, label='Both'):
    if os.path.exists(store_file):
        result_map = old_load_saved_result(store_file)
    else:
        result_map = {}

    result_map[label] = results

    output_results = open(store_file, 'wb')
    pickle.dump(result_map, output_results)
    output_results.close()


def store_intermediate_step(something, store_file):
    output = open(store_file, 'wb')
    pickle.dump(something, output)
    output.close()


def load_intermediate_step(store_file):
    input_hist = open(store_file, 'rb')
    something = pickle.load(input_hist)
    input_hist.close()
    # { 'label' : (acc, precision, recall, f1) }
    return something


def get_all_features(featureGenerators, dataset):
    features = []

    for user in sorted(dataset.keys()):
        # print (user)
        tweets = dataset[user].get_tweets()
        userFeatures = []
        for featureGenerator in featureGenerators:
            extracted = featureGenerator.extract_feature(user, tweets)
            if isinstance(extracted, np.ndarray):
                # print (extracted)
                userFeatures = userFeatures + extracted.tolist()
            else:
                userFeatures.append(extracted)
        features.append(userFeatures)
        # if len(userFeatures) > 1:
        #     print (userFeatures)
        #     features = features + userFeatures.tolist()
        # else:
        #     features.append(userFeatures)
    # features = np.array(features)
    # print (features)
    features = np.array(features)
    # normed_features = features / features.max(axis=0)
    # normed_features = (features - features.min(0)) / features.ptp(0)
    return features
    # return normed_features


from features.average_sentence_length_feature import AverageSentenceLengthFeature
from features.average_word_length_feature import AverageWordLengthFeature
from features.cap_sentence_feature import CapSentenceFeature
from features.cap_letters_feature import CapLettersFeature
from features.cap_words_feature import CapWordsFeature
from features.ends_with_interpunction_feature import EndsWithInterpunctionFeature
# from features.out_of_dict_words_feature import OutOfDictWordsFeature
from features.punctuation_count_feature import PunctuationCountFeature

# featureGenerators = [AverageSentenceLengthFeature(), AverageWordLengthFeature(), CapSentenceFeature()]

# featureGenerators = [CapSentenceFeature(),CapLettersFeature(),CapWordsFeature(),EndsWithInterpunctionFeature(),PunctuationCountFeature()]
#
# dataset = dr.load_dataset()
#
# print (getAllFeatures(featureGenerators, dataset)[:50])
