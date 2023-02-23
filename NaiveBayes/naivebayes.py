import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        return

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        self.p_spam = sum(y=='spam') / len(y) #empirical probability of ham_msg, spam_msg
        self.p_ham = sum(y=='ham') / len(y)
        msg_words = X.str.split() #split the messages into python lists of words
        lists = list(X.str.split())
        self.all_words = list(set([el for lst in lists for el in lst])) #build the word bag
        map = {word:{'ham':0,'spam':0} for word in self.all_words}

        for i in X.index:
            words = set(msg_words.loc[i]) #unique set of words within a message
            if y.loc[i] == 'ham':
                for word in words:
                    map[word]['ham'] += list(msg_words.loc[i]).count(word) #number of times the word appeared in the ham msg
            else:
                for word in words:
                    map[word]['spam'] += list(msg_words.loc[i]).count(word) #"..." spam msg

        self.data = pd.DataFrame(index=['ham','spam'],data=map)

    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        probs = []
        msg_words = X.str.split() #split the messages into python lists of words

        for i in X.index:
            unique_words = list(set(msg_words.loc[i])) #unique set of words for a given message
            word_counts = {word:0 for word in unique_words}
            for word in unique_words:
                word_counts[word] = msg_words.loc[i].count(word) #get frequency of each word
            #set up MLE estimation
            p_spam_msg = self.p_spam*np.product(np.array([(self.data.loc['spam',word] / self.data.loc['spam'].sum())**(word_counts[word]) for word in unique_words if word in self.all_words]))
            p_ham_msg = self.p_ham*np.product(np.array([(self.data.loc['ham',word] / self.data.loc['ham'].sum())**(word_counts[word]) for word in unique_words if word in self.all_words]))
            probs.append(np.array([p_ham_msg,p_spam_msg]))

        probs = np.array(probs)

        return probs

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        probs = self.predict_proba(X)

        classifications = np.array([np.argmax(probs[i]) for i in range(len(probs))]) #MLE estimation
        return ["ham" if maxdex==0 else "spam" for maxdex in classifications]

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        log_probs = []
        msg_words = X.str.split() #split the messages into python lists of words

        for i in X.index:
            unique_words = list(set(msg_words.loc[i])) #unique set of words for a given message
            word_counts = {word:0 for word in unique_words}
            for word in unique_words:
                word_counts[word] = msg_words.loc[i].count(word) #get frequency of each word
            #set up MLE estimation
            log_p_spam_msg = np.sum(np.array([(word_counts[word])*np.log((self.data.loc['spam',word]+1) / (self.data.loc['spam'].sum()+2)) for word in unique_words if word in self.all_words]))
            log_p_ham_msg = np.sum(np.array([(word_counts[word])*np.log((self.data.loc['ham',word]+1) / (self.data.loc['ham'].sum()+2)) for word in unique_words if word in self.all_words]))
            log_probs.append(np.array([log_p_ham_msg,log_p_spam_msg]))

        log_probs = np.array(log_probs)
        log_probs[:,0] += np.log(self.p_ham)
        log_probs[:,1] += np.log(self.p_spam)

        return log_probs


    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        log_probs = self.predict_log_proba(X)
        
        classifications = np.array([np.argmax(log_probs[i]) for i in range(len(log_probs))]) #MLE estimation
        return ["ham" if maxdex==0 else "spam" for maxdex in classifications]


class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables
    '''

    def __init__(self):
        return


    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels

        Returns:
            self: this is an optional method to train
        '''
        self.p_spam = sum(y=='spam') / len(y) #empirical probability of ham_msg, spam_msg
        self.p_ham = sum(y=='ham') / len(y)
        msg_words = X.str.split() #split the messages into python lists of words
        lists = list(X.str.split())
        self.all_words = list(set([el for lst in lists for el in lst])) #build word bag
        map = {word:{'ham':0,'spam':0} for word in self.all_words}

        for i in X.index:
            words = set(msg_words.loc[i]) #unique set of words within a message
            if y.loc[i] == 'ham':
                for word in words:
                    map[word]['ham'] += list(msg_words.loc[i]).count(word) #number of times the word appeared in the ham msg
            else:
                for word in words:
                    map[word]['spam'] += list(msg_words.loc[i]).count(word) #"..." spam msg

        self.data = pd.DataFrame(index=['ham','spam'],data=map) 

        self.ham_rates = dict((self.data.loc['ham']+1) / (self.data.loc['ham'].sum()+2))
        self.spam_rates = dict((self.data.loc['spam']+1) / (self.data.loc['spam'].sum()+2))

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam
                column 0 is ham, column 1 is spam
        '''
        log_probs = []
        msg_words = X.str.split() #split the messages into python lists of words

        for i in X.index:
            n = len(msg_words.loc[i]) #length of message
            unique_words = list(set(msg_words.loc[i])) #unique set of words for a given message
            word_counts = {word:0 for word in unique_words} 
            for word in unique_words:
                word_counts[word] = msg_words.loc[i].count(word) #get frequency of each word
            #set up MLE estimation
            log_poisson_ham = np.sum([np.log(stats.poisson.pmf(word_counts[word],self.ham_rates[word]*n)) for word in unique_words if word in self.all_words])
            log_poisson_spam = np.sum([np.log(stats.poisson.pmf(word_counts[word],self.spam_rates[word]*n)) for word in unique_words if word in self.all_words])
            log_probs.append(np.array([log_poisson_ham,log_poisson_spam]))
        
        log_probs = np.array(log_probs)

        log_probs[:,0] += np.log(self.p_ham)
        log_probs[:,1] += np.log(self.p_spam)

        return log_probs

    def predict(self, X):
        '''
        Use self.predict_log_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        log_probs = self.predict_log_proba(X)
        
        classifications = np.array([np.argmax(log_probs[i]) for i in range(len(log_probs))]) #MLE estimation
        return ["ham" if maxdex==0 else "spam" for maxdex in classifications]


def test_fit():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    naivebabe = NaiveBayesFilter()
    naivebabe.fit(X[:300],y[:300])
    assert naivebabe.data.loc['ham','in'] == 47
    assert naivebabe.data.loc['spam','in'] == 4

def test_predict_proba():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    naivebabe = NaiveBayesFilter()
    naivebabe.fit(X[:300],y[:300])
    probs = naivebabe.predict_proba(X[530:535])
    true_probs = np.array([[8.33609611e-16, 0.00000000e+00],
                            [0.00000000e+00, 2.26874221e-44],
                            [0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00],
                            [2.50103642e-10, 0.00000000e+00]])
    assert np.allclose(probs,true_probs)

def test_predict():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    naivebabe = NaiveBayesFilter()
    naivebabe.fit(X[:300],y[:300])
    preds = naivebabe.predict(X[530:535])
    truth = np.array(['ham', 'spam', 'ham', 'ham', 'ham'])
    assert (preds == truth).all()

def test_predict_log_proba():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    naivebabe = NaiveBayesFilter()
    naivebabe.fit(X[:300],y[:300])
    probs = naivebabe.predict_log_proba(X[530:535])
    true_probs = np.array([[ -33.39347149,  -35.34710583],
                            [-106.83571245,  -93.63509276],
                            [ -57.05676356,  -58.34010293],
                            [ -19.22723879,  -20.19409107],
                            [ -21.5513236 ,  -26.18555562]])
    assert np.allclose(probs,true_probs)

def test_predict_log():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    naivebabe = NaiveBayesFilter()
    naivebabe.fit(X[:300],y[:300])
    preds = naivebabe.predict_log(X[530:535])
    truth = np.array(['ham', 'spam', 'ham', 'ham', 'ham'])
    assert (preds == truth).all()

def test_poisson_fit():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    poissonbabe = PoissonBayesFilter()
    poissonbabe.fit(X[:300],y[:300])
    assert np.isclose(poissonbabe.ham_rates['in'], 0.012588512981904013)
    assert np.isclose(poissonbabe.spam_rates['in'],0.004166666666666667)

def test_poisson_predict_log_proba():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    poissonbabe = PoissonBayesFilter()
    poissonbabe.fit(X[:300],y[:300])
    probs = poissonbabe.predict_log_proba(X[530:535])
    true_probs = np.array([[-21.42246084, -23.29712325],
                            [-58.14578114, -44.50623148],
                            [-38.22508624, -39.48322892],
                            [-14.45137719, -15.419944  ],
                            [-16.23273939, -20.68704484]])
    assert np.allclose(probs,true_probs)

def test_poisson_predict():
    df = pd.read_csv("sms_spam_collection.csv")
    X, y = df.Message,df.Label
    poissonbabe = PoissonBayesFilter()
    poissonbabe.fit(X[:300],y[:300])
    preds = poissonbabe.predict(X[530:535])
    truth = np.array(['ham', 'spam', 'ham', 'ham', 'ham'])
    assert (preds == truth).all()



def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)

    clf = MultinomialNB() #multinomial naive bayes classifier
    clf = clf.fit(train_counts,y_train)

    test_counts = vectorizer.transform(X_test)
    labels = clf.predict(test_counts) #make predictions

    return labels

def test_sklearn_method():
    df = pd.read_csv('sms_spam_collection.csv')
    X, y = df['Message'],df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)


    actual_labels = sklearn_method(X_train, y_train, X_test)

    naivebabe = NaiveBayesFilter()
    naivebabe.fit(X_train, y_train)
    naivebabe_labels = naivebabe.predict_log(X_test)
    print(accuracy_score(actual_labels,naivebabe_labels))

    poissonbabe = PoissonBayesFilter()
    poissonbabe.fit(X_train,y_train)
    poissonbabe_labels = poissonbabe.predict(X_test)
    print(accuracy_score(actual_labels,poissonbabe_labels))

def test_all():
    test_fit()
    test_predict_proba()
    test_predict()
    test_predict_log_proba()
    test_predict_log()
    test_poisson_fit()
    test_poisson_predict_log_proba()
    test_poisson_predict()
    test_sklearn_method()
