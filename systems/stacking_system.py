from systems.system_evaluation import EvaluationSystem
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from evaluation.eval_utils import get_documents_y
from reduction.reduction_storage import ReductionWithStorage
from reduction.inform_gain_old import InformationGainOld
from reduction.reduction_storage import ReductionWithStorage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC


class StackingEvaluation(EvaluationSystem):
    def __init__(self, gender_stack=(MultinomialNB, BernoulliNB, LinearSVC, BayesianRidge), gender_meta=BernoulliNB,
                 age_stack=(MultinomialNB, LogisticRegression, BernoulliNB, LinearSVC), age_meta=LinearSVC, n_range=(1, 1)):
        self.gender_stack = gender_stack
        self.gender_meta = gender_meta
        self.age_stack = age_stack
        self.age_meta = age_meta
        self.n_range = n_range

    @staticmethod
    def space_splitter(sentence):
        return sentence.split(" ")

    def get_features(self, dataset, classification):
        documents, y = get_documents_y(dataset, classification)

        ## Definining tf-idf vector
        vectorizer = TfidfVectorizer(ngram_range=self.n_range, tokenizer=self.space_splitter)
        vectorizer.fit(documents)

        features = vectorizer.transform(documents)
        names = vectorizer.get_feature_names()

        return features, y, names

    def get_clf(self, classification):
        meta_G = None
        meta_A = None
        classifier = None

        if classification == "gender" or classification == "both":
            classifier = meta_G = StackingClassifier(
                classifiers=[clf() for clf in self.gender_stack],
                meta_classifier=self.gender_meta())
        if classification == "age" or classification == "both":
            classifier = meta_A = StackingClassifier(
                classifiers=[clf() for clf in self.age_stack],
                meta_classifier=self.age_meta())
        if classification == "both":
            classifier = VotingClassifier(estimators=[('meta_A', meta_A), ('meta_G', meta_G)])

        return 'clf', classifier

    @staticmethod
    def default_information_gain_reduce():
        return ReductionWithStorage(base_reduction=InformationGainOld(1e-3), base_clf=StackingEvaluation,
                                    threshold=1e-3)
