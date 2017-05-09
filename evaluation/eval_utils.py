import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle
import dataset.dataset_reader as dr
import numpy as np

def print_evaluation_results(pipeline, parameters, X, y, k1=10, k2=3):
    accuracy, precision, recall, f1 = nested_k_fold_cv(pipeline, parameters, X, y, k1=k1, k2=k2)
    # print(accuracy)
    # print(precision)
    # print(recall)
    # print(f1)
    print("accuracy,precisionMacro,recallMacro,f1Macro")
    print(np.average(accuracy), np.average(precision), np.average(recall), np.average(f1))


def nested_k_fold_cv(clf, param_grid, X, y, k1=10, k2=3):
    # kf1 = KFold(n_splits=k1)
    kf1 = StratifiedKFold(n_splits=k1,random_state=42)

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


def load_saved_result(path):
    input_hist = open(path, 'rb')
    results = pickle.load(input_hist)
    input_hist.close()
    # { 'label' : (acc, precision, recall, f1) }
    return results


def store_result(results, store_file, label='Both'):
    if os.path.exists(store_file):
        result_map = load_saved_result(store_file)
    else:
        result_map = {}

    result_map[label] = results

    output_results = open(store_file, 'wb')
    pickle.dump(result_map, output_results)
    output_results.close()

def getAllFeatures(featureGenerators,dataset):

    features = []

    for user in sorted(dataset.keys()):
        #print (user)
        tweets = dataset[user].get_tweets()
        userFeatures = []
        for featureGenerator in featureGenerators:
            extracted = featureGenerator.extract_feature(user,tweets)
            if isinstance(extracted, np.ndarray):
                #print (extracted)
                userFeatures = userFeatures + extracted.tolist()
            else:
                userFeatures.append(extracted)
        features.append(userFeatures)
        # if len(userFeatures) > 1:
        #     print (userFeatures)
        #     features = features + userFeatures.tolist()
        # else:
        #     features.append(userFeatures)
    #features = np.array(features)
    #print (features)
    features = np.array(features)
    normed_features = features / features.max(axis=0)
    #normed_features = (features - features.min(0)) / features.ptp(0)
    return normed_features

from features.average_sentence_length_feature import AverageSentenceLengthFeature
from features.average_word_length_feature import AverageWordLengthFeature
from features.cap_sentence_feature import CapSentenceFeature
from features.cap_letters_feature import CapLettersFeature
from features.cap_words_feature import CapWordsFeature
from features.ends_with_interpunction_feature import EndsWithInterpunctionFeature
#from features.out_of_dict_words_feature import OutOfDictWordsFeature
from features.punctuation_count_feature import PunctuationCountFeature

#featureGenerators = [AverageSentenceLengthFeature(), AverageWordLengthFeature(), CapSentenceFeature()]

# featureGenerators = [CapSentenceFeature(),CapLettersFeature(),CapWordsFeature(),EndsWithInterpunctionFeature(),PunctuationCountFeature()]
#
# dataset = dr.load_dataset()
#
# print (getAllFeatures(featureGenerators, dataset)[:50])
