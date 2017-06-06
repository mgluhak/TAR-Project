import pickle

# genderModel = pickle.load(open("gender.pkl", "rb"))
# ageModel = pickle.load(open("age.pkl", "rb"))

from demonstration.eval_utils import get_documents_y

def spaceSplitter(list):
    return list.split(" ")

vectorizer = pickle.load(open("tf-idf_vectorizer.pkl", "rb"))


def getFeatures(dataset, classification="both"):
    documents, y = get_documents_y(dataset, classification)
    features = vectorizer.transform(documents)
    return features, y

# from evaluation.eval_utils import get_all_features
# def selectFeatures(dataset):
#     from features.average_sentence_length_feature import AverageSentenceLengthFeature
#     from features.average_word_length_feature import AverageWordLengthFeature
#     from features.cap_sentence_feature import CapSentenceFeature
#     from features.cap_letters_feature import CapLettersFeature
#     from features.cap_words_feature import CapWordsFeature
#     from features.ends_with_interpunction_feature import EndsWithInterpunctionFeature
#     from features.out_of_dict_words_feature import OutOfDictWordsFeature
#     from features.punctuation_count_feature import PunctuationCountFeature
#     from features.pos_tags import PosTagFeature
#     from features.feature_storage import FeatureWithStorage
#
#     # featureGenerators = [AverageSentenceLengthFeature(), AverageWordLengthFeature(), CapSentenceFeature()]
#
#     featureGenerators = [CapSentenceFeature(), CapLettersFeature(), CapWordsFeature(), EndsWithInterpunctionFeature(),
#                          PunctuationCountFeature(),FeatureWithStorage(PosTagFeature(type="Perceptron"),'pos2.shelve')]
#
#     return get_all_features(featureGenerators, dataset)
#

# from scipy import sparse

# def getPredictedClasses(classification = "gender"):
#     dataset = dr.load_dataset()
#     baselineFeatures, y = getFeatures(dataset, classification)
#     otherFeatures = selectFeatures(dataset)
#
#     # print (baselineFeatures.shape, otherFeatures.shape)
#     # 4. stackanje
#     X = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(otherFeatures), baselineFeatures]))
#     # print (X.shape)
#     # X = np.hstack([baselineFeatures,otherFeatures])
#     # X = np.concatenate((baselineFeatures,otherFeatures),axis=1)
#     # print (X.shape)
#
#     if classification == "gender":
#         model = pickle.load(open("gender.pkl", "rb"))
#     elif classification == "age":
#         model = pickle.load(open("age.pkl", "rb"))
#
#     print (y[:10])
#     y = model.predict(X)
#     return y


from dataset.dataset_map_entry import *
def getTrumpFeatures(classification = "gender"):
    tweets = pickle.load(open("trump.pkl", "rb"))
    #tweets = tweets[500:2500]

    #tweets = dataset['b88171637fa04a302e94b14402f2793a'].get_tweets()

    documents = []
    document = []
    for tweet in tweets:
        document.append(" ".join(tweet))
    documents.append(" ".join(document))

    X = vectorizer.transform(documents)

    #trump_dataset = {}
    #trump_dataset[0] = TweetMapEntry(Gender.MALE, AgeGroup._65_xx, tweets[700:800])

    #otherFeatures = selectFeatures(trump_dataset)
    #X = sparse.csr_matrix(sparse.hstack([sparse.csr_matrix(otherFeatures), X]))

    return X

tweets = pickle.load(open("LeoDiCaprio.pkl", "rb"))
tweets = tweets[100:120] + tweets[500:]
from evaluation.eval_utils import store_intermediate_step
store_intermediate_step(tweets, "LeoDiCaprio.pkl")
# X = getTrumpFeatures()
#
#
# genderModel = pickle.load(open("gender.pkl", "rb"))
# ageModel = pickle.load(open("age.pkl", "rb"))
#
# print (genderModel.predict(X))
# print (ageModel.predict(X))


# yGender = getPredictedClasses(classification="gender")
# print (yGender[:10])
#
# yAge = getPredictedClasses(classification="age")
# print (yAge[:10])


# dataset = dr.load_dataset()
# # tweets = dataset['b88171637fa04a302e94b14402f2793a'].get_tweets()
#
# documents, y = get_documents_y(dataset, "age")
#
# ## Definining tf-idf vector
# vectorizer = TfidfVectorizer(tokenizer=spaceSplitter)
# vectorizer.fit(documents)
#
#
# output_file = open("tf-idf_vectorizer.pkl", "wb")
# pickle.dump(vectorizer, output_file)
# output_file.close()