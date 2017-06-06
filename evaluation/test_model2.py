from sklearn import preprocessing
import dataset.dataset_reader as dr
from evaluation.eval_utils import nested_k_fold_cv
from evaluation.eval_utils import old_store_result
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from evaluation.baseline1 import getFeatures
from evaluation.eval_utils import get_all_features
import numpy as np
from scipy import sparse

from metrics.metrics_map import MetricsMap


def selectFeatures(dataset):
    from features.average_sentence_length_feature import AverageSentenceLengthFeature
    from features.average_word_length_feature import AverageWordLengthFeature
    from features.cap_sentence_feature import CapSentenceFeature
    from features.cap_letters_feature import CapLettersFeature
    from features.cap_words_feature import CapWordsFeature
    from features.ends_with_interpunction_feature import EndsWithInterpunctionFeature
    from features.out_of_dict_words_feature import OutOfDictWordsFeature
    from features.punctuation_count_feature import PunctuationCountFeature
    from features.pos_tags import PosTagFeature
    from features.feature_storage import FeatureWithStorage

    # featureGenerators = [AverageSentenceLengthFeature(), AverageWordLengthFeature(), CapSentenceFeature()]

    featureGenerators = [CapSentenceFeature(), CapLettersFeature(), CapWordsFeature(), EndsWithInterpunctionFeature(),
                         PunctuationCountFeature(),FeatureWithStorage(PosTagFeature(type="Perceptron"),'cache/pos2.shelve')]

    return get_all_features(featureGenerators, dataset)

from features.glove_features import GloveFeatures
w2v = GloveFeatures()
# clasification - possible modes - age, gender, both
def evaluate(classification="both"):
    # 1. ucitavanje dataseta
    dataset = dr.load_dataset()
    # 2. dohvacanje featura
    baselineFeatures, y = getFeatures(dataset, classification)
    # 3. dohvacanje ostalih featura
    otherFeatures = selectFeatures(dataset)

    # print (baselineFeatures.shape, otherFeatures.shape)
    # 4. stackanje
    X = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(otherFeatures), baselineFeatures]))
    # print (X.shape)
    # X = np.hstack([baselineFeatures,otherFeatures])
    # X = np.concatenate((baselineFeatures,otherFeatures),axis=1)
    # print (X.shape)

    w2v_features = []
    for user in sorted(dataset.keys()):
        tweets = dataset[user].get_tweets()
        if len(tweets) == 0:
            w2v_features.append(np.ones(100))
            continue
        w2v_features.append(np.array(w2v.extract_feature(user, tweets)))

    w2v_features = np.array(w2v_features)
    # print (w2v_features.shape)

    X = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(w2v_features), X]))
    # features = scipy.sparse.hstack((features,w2v_features))
    # features = np.concatenate((features,w2v_features),axis=1)


    # 5. encoding
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)

    yLabel = encoder.transform(y)

    ## Training linear SVM

    pot2func = lambda x: 2 ** x
    pot2 = map(pot2func, range(-5, 5))
    param_grid = {'svc__C': list(pot2)}
    clfSVM = LinearSVC()
    # scaler = preprocessing.StandardScaler()
    pipeline = Pipeline([('svc', clfSVM)])

    ## K- fold validation
    # print_evaluation_results(pipeline, param_grid, X, yLabel, k1=5, k2=3)
    return nested_k_fold_cv(pipeline, param_grid, X, yLabel, k1=5, k2=3)


from evaluation.eval_utils import print_evaluation_results
# print_evaluation_results()
# evaluate("gender")

old_store_result((evaluate("gender")), 'results/baseline+features+w2v.pkl', "Gender only")
#old_store_result((evaluate("age")), 'results/baseline+features+w2v.pkl', "Age only")
#old_store_result((evaluate("both")), 'results/baseline+features+w2v.pkl', "Both")