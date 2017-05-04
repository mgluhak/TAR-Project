import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import StratifiedKFold

def evaluate(pipeline,parameters,X,y,k1=10,k2=3):
    accuracy,precision,recall,f1= nested_kfold_cv(pipeline,parameters,X,y,k1=k1,k2=k2)
    print (accuracy)
    print (precision)
    print (recall)
    print (f1)
    print ("accuracy,precisionMacro,recallMacro,f1Macro")
    print (np.average(accuracy),np.average(precision),np.average(recall),np.average(f1))



def nested_kfold_cv(clf, param_grid, X, y, k1=10, k2=3):
    #kf1 = KFold(n_splits=k1)
    kf1 = StratifiedKFold(n_splits=k1)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    counter = 1
    for train_index, test_index in kf1.split(X,y):
        print (counter)
        counter += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gs = GridSearchCV(clf,param_grid,cv=k2,n_jobs=1)
        gs.fit(X_train,y_train)
        #print gs.best_estimator_
        predictedValues = gs.predict(X_test)
        accuracy.append(accuracy_score(y_test,predictedValues))
        precision.append(precision_score(y_test,predictedValues,average="macro"))
        recall.append(recall_score(y_test,predictedValues,average="macro"))
        f1.append(f1_score(y_test,predictedValues,average="macro"))
    return (accuracy,precision,recall,f1)
