from metrics.metrics_map import MetricsMap
from metrics.standard_metrics import StandardMetrics
from systems.system_evaluation import EvaluationSystem
from evaluation.eval_utils import get_documents_y
from validation.k_fold import KFoldValidation
from vectorizer.tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from dataset import dataset_reader as dr
import nltk
import gensim
import os


class Word2VecEvaluation(EvaluationSystem):
    def __init__(self, pre_trained=None, clf=("extra trees", ExtraTreesClassifier(n_estimators=200)),
                 vectorizer_class=TfidfEmbeddingVectorizer):
        self.clf_ = clf
        self.pre_trained = pre_trained
        self.vectorizer_class = vectorizer_class
        self.vectorizer = None

    def get_features(self, dataset, classification):
        documents, y = get_documents_y(dataset, classification)
        if self.pre_trained is None:
            data_path_name = 'dataset'
            data = documents
        else:
            data_path_name = self.pre_trained[0]
            data = gensim.models.Word2Vec.load_word2vec_format(self.pre_trained[1], binary=True)

        tok_corp = [nltk.word_tokenize(doc) for doc in data]
        if os.path.exists('cache/' + str(data_path_name) + '_trained_model_' + str(classification) + '.pkl'):
            model = gensim.models.Word2Vec.load('cache/' + str(data_path_name) + '_trained_model_' + str(classification)+ '.pkl')
        else:
            # model = gensim.models.Word2Vec(tok_corp, min_count=1, size=32)
            model = gensim.models.Word2Vec(tok_corp, size=100, window=5, min_count=5)
            model.save('cache/' + str(data_path_name) + '_trained_model_' + str(classification) + '.pkl')
        print("w2v loaded...")
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        # print("w2v")

        self.vectorizer = self.vectorizer_class(w2v)

        self.vectorizer.fit(data, y)

        features = self.vectorizer.transform(X=data)

        return features, y, None

    def get_clf(self, classification):
        return self.clf_

# w2v = Word2VecEvaluation()
# kf = KFoldValidation(random_state=42)
# sm = StandardMetrics()
# mm = MetricsMap()
#
#
# mm.evaluate(dr.load_dataset(), w2v, kf, "both")