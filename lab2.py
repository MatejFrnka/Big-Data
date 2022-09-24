import pandas as pd
import scipy.optimize
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PROCESSED_TEXT = "processed_text"
PROCESSED_TEXT_AS_STRING = "processed_text_as_string"
VECTORIZED_TEXT = "vectorized_text"
SCORE = "score"
STOP_WORDS = stopwords.words('english')


# Removes extra whitespaces as well as leading/trailing whitespaces
def remove_whitespace(text):
    text.strip()
    return " ".join(text.split())


# Removes emojis (there are probably more efficient ways of doing this but I think this works)
def remove_emojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')


# Removes URLs, replaces with whitespace
def remove_URL(text):
    return re.sub(r"http\S+", " ", text)


# Removes html code, replaces with whitespace
def remove_html(text):
    return re.sub('<.*?>', " ", text)


# Removes numbers, replaces with whitespace
def remove_numbers(text):
    return re.sub("\d", " ", text)


# Removes punctuations and other special characters, replaces with whitespace
def remove_punctuation(text):
    return re.sub("[.,!?:;-='...\"@#_/]", " ", text)


# Removes stopwords. Can easily be tweaked to remove or add certain words
def remove_stopwords(text):
    words = text.split()
    stops_removed = []
    for word in words:
        if word not in STOP_WORDS:
            stops_removed.append(word)
    return " ".join(stops_removed)


# Lemmatizes verbs. Can be extended to match for other word types than verbs but run time would increase.
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for word in text:
        lemmatized.append(lemmatizer.lemmatize(word, 'v'))
    return lemmatized


# counts is a dictionary with key == unique word, value == number of occurences
def word_count(text, counts):
    for word in text:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


# Gets all unique words and stores them in a set
def get_words(text, wordset):
    return wordset.update(text)


def compile(array):
    return " ".join(array)


# Order of operations not completely arbitrary so there could be better ways to order the functions.
def preprocess(df):
    df[PROCESSED_TEXT] = df["text"].apply(remove_URL)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(remove_html)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].str.lower()
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(remove_punctuation)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(remove_numbers)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(remove_emojis)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(remove_stopwords)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(remove_whitespace)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(word_tokenize)
    df[PROCESSED_TEXT] = df[PROCESSED_TEXT].apply(lemmatize_text)
    df[PROCESSED_TEXT_AS_STRING] = df[PROCESSED_TEXT].apply(compile)
    df[SCORE] = df[SCORE].apply(lambda x: 1 if x == 1 else -1)


# Different kernel implementations
class KernelLinear:
    def kernel(self, x1, x2):
        return np.dot(x1, x2.T)


class KernelRFB:
    def __int__(self, gamma):
        self.gamma = gamma

    def kernel(self, x1, x2):
        return np.exp(-self.gamma * np.sum((x2 - x1[:, np.newaxis]) ** 2, axis=-1))


class KernelPolynomial:
    def __int__(self, degree):
        self.degree = degree

    def kernel(self, x1, x2):
        return np.dot(x1, x2.T) ** self.degree


class CustomSVM:
    def __init__(self, kernel, upper_bound):
        self.upper_bound = upper_bound
        self.kernel = kernel
        self.b = None
        self.non_zero_alpha = None

    def fit(self, data, validation):
        N = len(data)
        t1 = time.time()
        # Precomputing what can be precomputed so that the function objective takes as short as possible
        precomputed_array = np.empty([N, N])
        for i in range(N):
            for j in range(N):
                precomputed_array[i][j] = validation[i] * validation[j] * self.kernel(data[i], data[j])
        print(f"Precomputed array in: {time.time() - t1}s")

        # This function is called by the optimisation library scipy.
        # The library is trying to get this to be as close to 0 under the given constraints
        # Performance is crucial here, this function is called thousands of times
        # The alpha value is a vector of length N
        def objective(alpha):
            return 0.5 * alpha.dot(alpha.dot(precomputed_array)) - alpha.sum()

        # This is one of the constraints
        # The minimisation library must always select alpha so that this function is equal to 0
        def zerofun(alpha):
            return np.dot(alpha, validation)

        # This is another constraint, it specifies the range in which alpha_i can be.
        # If upper_bound is None, there is no upper limit for alpha_i
        bounds = [(0, self.upper_bound) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': zerofun}
        t2 = time.time()
        # Optimization function
        ret = scipy.optimize.minimize(objective, np.zeros(N), bounds=bounds, constraints=constraint)
        print(f"Optimized in: {time.time() - t2}s")
        estimated_alpha = ret['x']
        self.non_zero_alpha = []
        # Select non zero alpha_i
        # Epsilon = 10^-5
        for i in range(len(estimated_alpha)):
            if estimated_alpha[i] > 1 / 10 ** 5:
                self.non_zero_alpha.append((estimated_alpha[i], data[i], validation[i]))
        # B is te bias of the model, it is calculated here
        self.b = 0
        for i in range(len(estimated_alpha)):
            self.b += estimated_alpha[i] * validation[i] * self.kernel(self.non_zero_alpha[0][1], data[i])
        print(self.b)

    def predict_iterable(self, vectors):
        return [self.predict(vector) for vector in vectors]

    def predict(self, vector):
        acc = 0
        for val in self.non_zero_alpha:
            acc += val[0] * val[2] * self.kernel(vector, val[1])
        return -1 if acc < 0 else 1


traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('train.csv')
evaldata = pd.read_csv('evaluation.csv')

t1 = time.time()
preprocess(traindata)
preprocess(testdata)
preprocess(evaldata)

print(f"Data processed: {time.time() - t1}s")
# Select 2000 samples from data
traindata = traindata.sample(2000)
tfidf_vect = TfidfVectorizer(max_features=400)
tfidf_vect.fit(traindata[PROCESSED_TEXT_AS_STRING])
tfid_train = tfidf_vect.transform(traindata[PROCESSED_TEXT_AS_STRING])
tfid_test = tfidf_vect.transform(testdata[PROCESSED_TEXT_AS_STRING])
tfid_eval = tfidf_vect.transform(evaldata[PROCESSED_TEXT_AS_STRING])

svm = CustomSVM(KernelLinear().kernel, 5)

# Training of the model is very slow
svm.fit(tfid_train.toarray(), traindata[SCORE].to_numpy())
pred_train = svm.predict_iterable(tfid_test.toarray())
# But predicting is very fast
print("Train accuracy: ", accuracy_score(testdata[SCORE], pred_train) * 100)

pred_eval = svm.predict_iterable(tfid_eval.toarray())
print("Eval accuracy: ", accuracy_score(evaldata[SCORE], pred_eval) * 100)
