import dataset.dataset_reader as dr
import os
from evaluation.eval_utils import nested_k_fold_cv
from evaluation.eval_utils import old_store_result
from dataset.dataset_map_entry import AgeGroup
from dataset.dataset_map_entry import Gender
from evaluation.eval_utils import k_fold
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from evaluation.eval_utils import get_documents_y
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
from evaluation.eval_utils import store_intermediate_step
from evaluation.eval_utils import load_intermediate_step
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection
import pandas as pd


# custom spliter used instead of a tokenizer, since the tweets are already tokenized
def spaceSplitter(list):
    return list.split(" ")


def getFeatures(dataset, classification="both"):
    documents, y = get_documents_y(dataset, classification)

    ## Definining tf-idf vector

    vectorizer = TfidfVectorizer(tokenizer=spaceSplitter)
    vectorizer.fit(documents)

    features = vectorizer.transform(documents)
    names = vectorizer.get_feature_names()
    # print(len(names))
    # print(len(y))
    # print(len(documents))
    # print(len(features.todense()))
    return features, y, names


def information_gain(X, y):
    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
        for c in classCnt:
            probs = classCnt[c] / float(featureTot)
            entropy_x_set = entropy_x_set - probs * np.log(probs)
            probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
            entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        for c in classTotCnt:
            if c not in classCnt:
                probs = classTotCnt[c] / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                                 + ((tot - featureTot) / float(tot)) * entropy_x_not_set)

    tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1
    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log(probs)

    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain = []
    for i in range(0, len(nz[0])):
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre + 1, nz[0][i]):
                information_gain.append(0)
            ig = _calIg()
            information_gain.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1
    ig = _calIg()
    information_gain.append(ig)

    return np.asarray(information_gain)


# def information_gain(x, y):
#
#     def _entropy(values):
#         counts = np.bincount(values)
#         probs = counts[np.nonzero(counts)] / float(len(values))
#         return - np.sum(probs * np.log(probs))
#
#     def _information_gain(feature, y):
#         feature_set_indices = np.nonzero(feature)[1]
#         feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]
#         entropy_x_set = _entropy(y[feature_set_indices])
#         entropy_x_not_set = _entropy(y[feature_not_set_indices])
#
#         return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
#                                  + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))
#
#     feature_size = x.shape[0]
#     feature_range = range(0, feature_size)
#     entropy_before = _entropy(y)
#     information_gain_scores = []
#
#     for feature in x.T:
#         information_gain_scores.append(_information_gain(feature, y))
#     return information_gain_scores, []


# clasification - possible modes - age, gender, both
def evaluate(classification="both", load_stored=True, store_new=True):
    features, y, names = getFeatures(dataset=dr.load_dataset(), classification=classification)

    if load_stored and os.path.exists(os.getcwd() + '/cache/' + classification + '_features.pkl'):
        features = load_intermediate_step('cache/' + classification + '_features.pkl')
    else:
        reduced_features, new_names = filter_features_inf_gain(features, y, names, 1E-3)
        features = reduced_features
        if store_new:
            store_intermediate_step(reduced_features, 'cache/' + classification + '_features.pkl')

    # FEATURE TO DENSE !!!!!
    features = features.todense()
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)
    yLabel = encoder.transform(y)

    ## Training linear SVM

    # pot2func = lambda x: 2 ** x
    # pot2 = map(pot2func, range(-5, 5))
    # param_grid = {'svc__C': list(pot2)}
    # clfSVM = LinearSVC()
    # pipeline = Pipeline([('svc', clfSVM)])
    #
    # ## K- fold validation
    # return nested_k_fold_cv(pipeline, param_grid, features, yLabel, k1=5, k2=3)
    meta_G = None
    meta_A = None
    classifier = None
    if classification == "gender" or classification == "both":
        naive_bayes_multi = MultinomialNB()
        naive_bayes = BernoulliNB()
        linear_svm = LinearSVC()
        bayes_logistic = BayesianRidge()
        classifier = meta_G = StackingClassifier(
            classifiers=[naive_bayes_multi, naive_bayes, linear_svm, bayes_logistic],
            meta_classifier=BernoulliNB())
    if classification == "age" or classification == "both":
        naive_bayes_multi = MultinomialNB()
        simple_log = LogisticRegression()
        naive_bayes = BernoulliNB()
        linear_svm = LinearSVC()
        classifier = meta_A = StackingClassifier(classifiers=[naive_bayes_multi, simple_log, naive_bayes, linear_svm],
                                                 meta_classifier=LinearSVC())
    if classification == "both":
        classifier = VotingClassifier(estimators=[('meta_A', meta_A), ('meta_G', meta_G)])

    #param_grid = {}
    #pipeline = Pipeline([('clf', classifier)])

    #return nested_k_fold_cv(pipeline, param_grid, features, yLabel, k1=5, k2=3)
    return k_fold(classifier, features, yLabel, k=5)


def filter_features_inf_gain(features, y, names, threshold, count_nan=True):
    new_names = []
    # DEBUG
    f = features.tolil()
    counter = 0
    total = len(names)
    for ig, name in zip(information_gain(features.todense(), y), names):
        if (ig > threshold) or (count_nan and np.isnan(ig)):
            new_names.append(name)
        else:
            column = names.index(name)
            # uklanjanje
            f[:, column] = 0
        counter += 1
        if counter % 100 == 0:
            print(str(counter) + "/" + str(total))
    return f, np.array([new_names])


# kept = 0
# not_kept = 0
# total = 0
# nan = 0
# min = 1111
# #for feature, y_f in zip(features.todense(), y):
# igs = information_gain(features.todense(), y)
# print(len(igs))
# for ig in igs:
#     kept += 1 if (ig > 1E-2) else 0
#     not_kept += 1 if ig <= 0 else 0
#     total += 1
#     nan += 1 if np.isnan(ig) else 0
#     if ig < min:
#         min = ig
#
# print(min)
# #filtered = filter_features_inf_gain(features, y, 1E-2)
#
# print(str(kept + nan) + "/" + str(total))
# print(str(not_kept))
# print(str(nan))
# new_f, new_n = filter_features_inf_gain(features, y, names, 1E-2)
# print(len(new_n))
# print(str(filtered.shape))
# print(str(features.getnnz()))
#
# dtc = DecisionTreeClassifier()
# dtc.fit(features, y)
# print(str(dtc.tree_.best_error[1]))
# jvm.start()
# loader = Loader("weka.core.converters.ArffLoader")
# anneal_data = loader.load_file('/home/mihael/Documents/8. semestar/APT/Projekt/bas pravi git/TAR-Project/dataset/output/age_features.arff')
# anneal_data.class_is_last()
#
# search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
# evaluation = ASEvaluation("weka.attributeSelection.InfoGainAttributeEval")
# attsel = AttributeSelection()
# attsel.ranking(True)
# attsel.folds(3)
# attsel.crossvalidation(True)
# attsel.search(search)
# attsel.evaluator(evaluation)
# attsel.select_attributes(anneal_data)
# print("ranked attrib)utes:\n" + str(attsel.ranked_attributes))
# print("result string:\n" + attsel.results_string)
#features, y, names = getFeatures(dataset=dr.load_dataset(), classification='age')
#reduced_features, new_names = filter_features_inf_gain(features, y, names, 1E-2)
#store_intermediate_step(reduced_features, 'cache/age_features.pkl')

old_store_result(evaluate("both", load_stored=True, store_new=True), 'results/stacking_17_5_21_55.pkl', "Both")
