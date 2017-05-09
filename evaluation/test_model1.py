from sklearn import preprocessing
import dataset.dataset_reader as dr
from evaluation.eval_utils import nested_k_fold_cv
from evaluation.eval_utils import store_result
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from evaluation.baseline1 import getFeatures
from evaluation.eval_utils import getAllFeatures
import numpy as np
from scipy import sparse

def selectFeatures(dataset):
    from features.average_sentence_length_feature import AverageSentenceLengthFeature
    from features.average_word_length_feature import AverageWordLengthFeature
    from features.cap_sentence_feature import CapSentenceFeature
    from features.cap_letters_feature import CapLettersFeature
    from features.cap_words_feature import CapWordsFeature
    from features.ends_with_interpunction_feature import EndsWithInterpunctionFeature
    from features.out_of_dict_words_feature import OutOfDictWordsFeature
    from features.punctuation_count_feature import PunctuationCountFeature

    #featureGenerators = [AverageSentenceLengthFeature(), AverageWordLengthFeature(), CapSentenceFeature()]

    featureGenerators = [CapSentenceFeature(),CapLettersFeature(),CapWordsFeature(),EndsWithInterpunctionFeature(),PunctuationCountFeature()]

    return getAllFeatures(featureGenerators, dataset)

# clasification - possible modes - age, gender, both
def evaluate(classification="both"):
    dataset = dr.load_dataset()
    baselineFeatures,y = getFeatures(dataset,classification)
    otherFeatures = selectFeatures(dataset)

    #print (baselineFeatures.shape, otherFeatures.shape)
    X = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(otherFeatures), baselineFeatures]))
    #print (X.shape)
    #X = np.hstack([baselineFeatures,otherFeatures])
    #X = np.concatenate((baselineFeatures,otherFeatures),axis=1)
    #print (X.shape)

    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)

    yLabel = encoder.transform(y)

    ## Training linear SVM

    pot2func = lambda x: 2 ** x
    pot2 = map(pot2func, range(-5, 5))
    param_grid = {'svc__C': list(pot2)}
    clfSVM = LinearSVC()
    #scaler = preprocessing.StandardScaler()
    pipeline = Pipeline([('svc', clfSVM)])

    ## K- fold validation
    #print_evaluation_results(pipeline, param_grid, X, yLabel, k1=5, k2=3)
    return nested_k_fold_cv(pipeline, param_grid, X, yLabel, k1=5, k2=3)

from evaluation.eval_utils import print_evaluation_results
#print_evaluation_results()
#evaluate("gender")

# store_result((evaluate("gender")), 'results/model1_svm_9_5_2017.pkl', "Gender only")
# store_result((evaluate("age")), 'results/model1_svm_9_5_2017.pkl', "Age only")
# store_result((evaluate("both")), 'results/model1_svm_9_5_2017.pkl', "Both")