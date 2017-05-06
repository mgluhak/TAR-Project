import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle


# def evaluate(pipeline, parameters, X, y, k1=10, k2=3):
# accuracy, precision, recall, f1 = nested_kfold_cv(pipeline, parameters, X, y, k1=k1, k2=k2)
# print(accuracy)
# print(precision)
# print(recall)
# print(f1)
# print("accuracy,precisionMacro,recallMacro,f1Macro")
# print(np.average(accuracy), np.average(precision), np.average(recall), np.average(f1))


def nested_k_fold_cv(clf, param_grid, X, y, k1=10, k2=3):
    # kf1 = KFold(n_splits=k1)
    kf1 = StratifiedKFold(n_splits=k1)

    accuracy = []
    precision = []
    recall = []
    f1 = []

    counter = 1  # DEBUG
    for train_index, test_index in kf1.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        gs = GridSearchCV(clf, param_grid, cv=k2, n_jobs=1)
        gs.fit(X_train, y_train)

        # print gs.best_estimator_
        y_predicted = gs.predict(X_test)

        print("iteration = " + str(counter) + "/" + str(k1))  # DEBUG
        counter += 1

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
