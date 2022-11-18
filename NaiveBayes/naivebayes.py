import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



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
        X['msg_words'] = X['Message'].str.split() #add a column for the words in each message
        lists = list(X['Message'].str.split())
        all_words = list(set([el for lst in lists for el in lst])) #word bag
        map = {word:{'ham':0,'spam':0} for word in all_words}

        for i in range(len(X)):
                for word in df.loc[i,'msg_words']:
                    if y[i] == 'ham':
                        map[word]['ham'] = list(X.loc[i,'msg_words']).count(word)
                    else:
                        map[word]['spam'] = list(X.loc[i,'msg_words']).count(word)


        NB = pd.DataFrame(index=['ham','spam']data=map)
        return NB

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

        raise NotImplementedError('Problem 2 incomplete')

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''

        raise NotImplementedError('Problem 3 incomplete')

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

        raise NotImplementedError('Problem 4 incomplete')


    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''

        raise NotImplementedError('Problem 4 incomplete')


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

        raise NotImplementedError('Problem 6 incomplete')

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam
                column 0 is ham, column 1 is spam
        '''

        raise NotImplementedError('Problem 7 incomplete')

    def predict(self, X):
        '''
        Use self.predict_log_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''

        raise NotImplementedError('Problem 7 incomplete')



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

    raise NotImplementedError('Problem 8 incomplete')


# class GNBC():
#     def __init__(self,eps):
#         """
#         parameters
#         ----------
#         eps: (float)
#         the minimum variance
#         """
#         self.eps = eps

#     #code
#     def fit(self,X,y):
#         """
#         page 274

#         parameters
#         ---------- 
#         X: (N,n_features) numpy array
#         data
#         y: (,N) numpy array
#         target

#         returns
#         -------
#         piss: (,n_classes) numpy array
#         probabilities of each class

#         probs:  
#         """
#         self.N, self.n_features = X.shape[0], X.shape[1]
#         if self.N != len(y):
#           raise ValueError("dimension mismatch")

#         self.classes = list(set(y)) #unqiue list
#         self.n_classes = len(self.classes)

#         self.class_counts = [] #count total occurences of each class
#         class_masks = [] #we will need to be able index into our data
#         for label in self.classes:
#           self.class_counts.append(sum(self.y == label)) #count the True's
#           class_masks.append(self.y == label)

#         self.class_counts = np.array(self._class_counts) #make it a numpy array

#         self.pis = self.class_counts / self.N #Nc/N

#         #iterate over i such that y_i==c (p. 274). this means we use class_masks
#         self.mus = np.array([[np.mean(X[class_masks[i],j] == label) for i in range(self.n_classes)] for j in range(self.n_features)])
#         self.ligmas = np.array([[np.sqrt(np.var(X[class_masks[i],j] == label)) for i in range(self.n_classes)] for j in range(self.n_features)])

#         #implement minimum variance
#         self.ligmas[np.allclose(self.ligmas,np.zeros_like(self.ligmas))] == self.eps

#     def predict(self,X):
#       """
#       parameters
#       ----------
#       X: (M,n_features) numpy array
#       data
#       returns
#       -------
#       the most likely class
#       """

#       M = X.shape[0]

#       #compute joint probability of data given theta
#       prior_probs = [np.product([np.product(norm.pdf(X[:,j],self.mus[j,k],self.ligmas[j,k])) for j in self.n_features]) for k in range(len(self.classes))]
      
#       output_probs = self.pis*prior_probs #compute probability of data given class c, theta
#       maxdex = np.argmax(output_probs) #index of most likely class

#       return self.classes[maxdex]