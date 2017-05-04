import dataset.dataset_reader as dr

#clasification - possible modes - age, gender, both
def evaluate(clasification="both"):
    dataset = dr.load_dataset()

    documents = []
    y = []

    #joining tokens with whitespace in order to fit the tf-idf vectorizer
    for user in dataset:
        tweets = dataset[user].get_tweets()
        if clasification == "both":
            y.append(str(dataset[user].get_gender().value)+str(dataset[user].get_age_group().value))
        elif clasification == "gender":
            y.append(str(dataset[user].get_gender().value))
        elif clasification == "age":
            y.append(str(dataset[user].get_age_group().value))
        else:
            raise ValueError("Given clasification taks is not specified")

        document = []
        for tweet in tweets:
            document.append(" ".join(tweet))
        documents.append(" ".join(document))

    from sklearn.feature_extraction.text import TfidfVectorizer

    #custom spliter used instead of a tokenizer, since the tweets are already tokenized
    def spaceSplitter(list):
        return list.split(" ")

    vectorizer = TfidfVectorizer(tokenizer=spaceSplitter)
    vectorizer.fit(documents)

    features = vectorizer.transform(documents)

    from sklearn import preprocessing
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)

    yLabel = encoder.transform(y)

    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    pot2func = lambda x: 2 ** x
    pot2 = map(pot2func, range(-5, 5))
    param_grid = {'svc__C': list(pot2)}
    clfSVM = LinearSVC()
    pipeline = Pipeline([('svc', clfSVM)])

    from evaluation.nestedKFold import evaluate
    evaluate(pipeline, param_grid, features, yLabel, k1=10, k2=3)

#evaluation results

#age only
#accuracy,precisionMacro,recallMacro,f1Macro
#0.451974730696 0.217529421246 0.256729323308 0.216938342466

#gender only
#accuracy,precisionMacro,recallMacro,f1Macro
#0.717965367965 0.720882394348 0.717965367965 0.717075870313

#age & gender
#accuracy,precisionMacro,recallMacro,f1Macro
#0.318092414832 0.237126847568 0.22628968254 0.213770186078
