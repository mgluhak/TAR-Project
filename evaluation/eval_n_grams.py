from validation.nested_k_fold import NestedKFoldValidation
from validation.k_fold import KFoldValidation
from metrics.standard_metrics import StandardMetrics
from metrics.metrics_map import MetricsMap
from systems.simple_system import SimpleEvaluation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest
from metrics.t_test_metrics import TTestMetrics
i

BOTH = "both"
AGE_ONLY = "age"
GENDER_ONLY = "gender"

SVM_LINEAR = 'svm_linear.pkl'
SVM_LINEAR_FEATURES = 'svm_linear_features.pkl'
SVM_LINEAR_SCALED_PCA = 'svm_linear_scl_pca.pkl'
SVM_LINEAR_FEATURES_CHI2 = 'svm_linear_feat_chi2.pkl'
STACKING = 'stacking.pkl'
WORD_2_VEC = 'word2vec.pkl'
WORD_2_VEC_GOOGLE = 'word2vec_google.pkl'
WORD_2_VEC_SVM = 'word2vec_svm.pkl'
LOGISTIC_REGRESSION = 'logistic_regression.pkl'
LOGISTIC_REGRESSION_FEATURES = 'logistic_regression_feature.pkl'

dataset = dr
se = SimpleEvaluation(n_gram_range=(1,3)) # Linear svm is default classifier
nkf = NestedKFoldValidation(param_grid=se.default_svm_get_param_grid(), random_state=42)
sm = StandardMetrics()
mm = MetricsMap()

mm.evaluate(dataset, se, nkf, BOTH)