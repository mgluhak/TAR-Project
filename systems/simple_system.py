from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from evaluation.eval_utils import get_documents_y
from systems.system_evaluation import EvaluationSystem
from features.average_sentence_length_feature import AverageSentenceLengthFeature
from features.average_word_length_feature import AverageWordLengthFeature
from features.cap_sentence_feature import CapSentenceFeature
from features.cap_letters_feature import CapLettersFeature
from features.cap_words_feature import CapWordsFeature
from features.ends_with_interpunction_feature import EndsWithInterpunctionFeature
from features.out_of_dict_words_feature import OutOfDictWordsFeature
from features.punctuation_count_feature import PunctuationCountFeature


class SimpleEvaluation(EvaluationSystem):
    def __init__(self, clf=('svc', LinearSVC())):
        self.clf_ = clf

    @staticmethod
    def space_splitter(sentence):
        return sentence.split(" ")

    def get_features(self, dataset, classification):
        documents, y = get_documents_y(dataset, classification)

        ## Definining tf-idf vector
        vectorizer = TfidfVectorizer(tokenizer=self.space_splitter)
        vectorizer.fit(documents)

        features = vectorizer.transform(documents)
        names = vectorizer.get_feature_names()

        return features, y, names

    def get_clf(self, classification):
        return self.clf_

    @staticmethod
    def default_svm_get_param_grid():
        return {'svc__C': list(map(lambda x: 2 ** x, range(-5, 5)))}

    @staticmethod
    def default_feature_set():
        return [CapSentenceFeature(), CapLettersFeature(), CapWordsFeature(), EndsWithInterpunctionFeature(),
                PunctuationCountFeature()]
