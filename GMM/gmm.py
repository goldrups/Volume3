# solutions.py
"""Volume 3: Gaussian Mixture Models. Solutions File."""

import numpy as np
from scipy import stats as st
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn import mixture, cluster

from sklearn.metrics import confusion_matrix

class GMM:
    # Problem 1
    def __init__(self, n_components, weights=None, means=None, covars=None):
        """
        Initializes a GMM.
        
        The parameters weights, means, and covars are optional. If fit() is called,
        they will be automatically initialized from the data.
        
        If specified, the parameters should have the following shapes, where d is
        the dimension of the GMM:
            weights: (n_components,)
            means: (n_components, d)
            covars: (n_components, d, d)
        """
        self.n_components = n_components
        if weights is not None:
            self.weights = weights
        else:
            self.weights = None
        if means is not None:
            self.means = means
        else:
            self.means = None
        if covars is not None:
            self.covars = covars
        else:
            self.covars = None

    
    # Problem 2
    def component_logpdf(self, k, z):
        """
        Returns the logarithm of the component pdf. This is used in several computations
        in other functions.
        
        Parameters:
            k (int) - the index of the component
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the log pdf of the component at 
        """
        w,mean,cov = self.weights[k],self.means[k],self.covars[k] #the parameters for the kth component
        return np.log(w) + multivariate_normal.logpdf(z,mean,cov)        
    
    # Problem 2
    def pdf(self, z):
        """
        Returns the probability density of the GMM at the given point or points.
        
        Parameters:
            z ((d,) or (..., d) ndarray) - the point or peints at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the GMM pdf at z
        """
        #calculate density
        return np.sum([self.weights[i]*multivariate_normal.pdf(z,mean=self.means[i],cov=self.covars[i]) for i in range(self.n_components)])
    

    # Problem 3
    def draw(self, n):
        """
        Draws n points from the GMM.
        
        Parameters:
            n (int) - the number of points to draw
        Returns:
            ((n,d) ndarray) - the drawn points, where d is the dimension of the GMM.
        """
        windex = np.random.choice(list(range(len(self.weights))),p=self.weights,size=n) #weight indices
        return np.array([np.random.multivariate_normal(self.means[windex[i]],self.covars[windex[i]]) for i in range(n)])
    
    # Problem 4
    def _compute_e_step(self, Z):
        """
        Computes the values of q_i^t(k) for the given data and current parameters.
        
        Parameters:
            Z ((n, d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components, n) ndarray): an array of the computed q_i^t(k) values, such
                    that result[k,i] = q_i^t(k).
        """
        n,d = Z.shape #n is number of data points living in R^d
        L = np.array([self.component_logpdf(k,Z) for k in range(self.n_components)])
        Li = L.max(axis=0) #max and sum along 0 axis
        Lexp = np.exp(L)
        Lexpli = Lexp * np.exp(-Li) 
        colsums = Lexpli.sum(axis=0)
        Q = Lexpli / colsums
        return Q

    # Problem 5
    def _compute_m_step(self, Z):
        """
        Takes a step of the expectation maximization (EM) algorithm. Return
        the updated parameters.
        
        Parameters:
            Z (n,d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components,) ndarray): the updated component weights
            ((n_components,d) ndarray): the updated component means
            ((n_components,d,d) ndarray): the updated component covariance matrices
        """
        n,d = Z.shape
        Q = self._compute_e_step(Z)
        new_weights = np.mean(Q,axis=1)
        new_means = Q@Z/np.sum(Q, axis=1).reshape(-1,1)

        centered = np.expand_dims(Z,0) - np.expand_dims(new_means,1) #einstein summation method
        new_covar = np.einsum("Kn,Knd,KnD -> KdD",Q,centered,centered)/np.sum(Q,axis=1).reshape(-1,1,1)

        return new_weights,new_means,new_covar
        
    # Problem 6
    def fit(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model by applying the Expectation Maximization algorithm until the
        parameters appear to converge.
        
        Parameters:
            Z ((n,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            self
        """
        n,d = Z.shape
        if self.weights == None:
            self.weights = np.ones(self.n_components) / self.n_components
        if self.means == None:
            self.means = np.array([Z[np.random.choice(len(Z))] for i in range(self.n_components)])
        if self.covars == None:
            self.covars = np.array([np.eye(d)*np.var(Z,axis=0) for i in range(self.n_components)])

        new_weights, new_means, new_covars = self._compute_m_step(Z)

        change = (np.max(np.abs(new_weights - self.weights)) + np.max(np.abs(new_means - self.means)) + np.max(np.abs(new_covars - self.covars)))

        while change > tol: #do this until we satisfy a convergence tolerance
            self.weights = new_weights
            self.means = new_means
            self.covars = new_covars

            new_weights, new_means, new_covars = self._compute_m_step(Z)

            change = (np.max(np.abs(new_weights - self.weights)) + np.max(np.abs(new_means - self.means)) + np.max(np.abs(new_covars - self.covars)))

        self.weights = new_weights
        self.means = new_means
        self.covars = new_covars
    
        return self
                    
    # Problem 8
    def predict(self, Z):
        """
        Predicts the labels of data points using the trained component parameters.
        
        Parameters:
            Z ((m,d) ndarray): the data to label; d is the dimension of the data.
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """
        return np.argmax([np.exp(self.component_logpdf(k,Z)) for k in range(self.n_components)],axis=0) #MLE
        
    def fit_predict(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model and predicts cluster labels.
        
        Parameters:
            Z ((m,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """
        return self.fit(Z, tol, maxiter).predict(Z)

# Problem 3
def problem3():
    """
    Draw a sample of 10,000 points from the GMM defined in the lab pdf. Plot a heatmap
    of the pdf of the GMM (using plt.pcolormesh) and a hexbin plot of the drawn points.
    How do the plots compare?
    """
    weights = np.array([0.6, 0.4])
    means = np.array([[-0.5, -4.0], [0.5, 0.5]])
    covars = np.array([[[1, 0],[0, 1]],[[0.25, -1],[-1, 8]]])
    gmm = GMM(n_components=2,weights=weights,means=means,covars=covars)

    x = np.linspace(-8,8,100)
    y = np.linspace(-8,8,100)
    X,Y = np.meshgrid(x,y)
    Z = np.array([[gmm.pdf([X[i,j],Y[i,j]]) for j in range(100)] for i in range(100)])
    plt.subplot(121).pcolormesh(X,Y,Z,shading='auto') #density

    n = 10**5
    D = gmm.draw(n)
    plt.subplot(122).hexbin(D[:,0],D[:,1]) #hexbin
    plt.show()
    
    

# Problem 7
def problem7(filename='problem7.npy'):
    """
    The file problem7.npy contains a collection of data drawn from a GMM.
    Train a GMM on this data with n_components=3. Plot the pdf of your
    trained GMM, as well as a hexbin plot of the data.
    """
    n_components = 3
    Z = np.load("problem7.npy") #load in data
    gmm = GMM(n_components=n_components)
    start = time.time() #timing
    gmm.fit(Z)
    finish = time.time()
    print("time: ", finish-start)

    x = np.linspace(-4,4,100)
    y = np.linspace(-4,4,100)
    X,Y = np.meshgrid(x,y)
    Z = np.array([[gmm.pdf([X[i,j],Y[i,j]]) for j in range(100)] for i in range(100)]) #pdf

    plt.subplot(121)
    plt.pcolormesh(X,Y,Z,shading='auto')
    plt.title('PDF')
    D = gmm.draw(10**5)
    plt.title("Draws")
    plt.subplot(122).hexbin(D[:,0],D[:,1],gridsize=(30,30)) #hexbin
    plt.xlim((-4,4))
    plt.ylim((-4,4))

    plt.show()


# Problem 8
def get_accuracy(pred_y, true_y):
    """
    Helper function to calculate the actually clustering accuracy, accounting for
    the possibility that labels are permuted.
    
    This computes the confusion matrix and uses scipy's implementation of the Hungarian
    Algorithm (linear_sum_assignment) to find the best combination, which is generally
    much faster than directly checking the permutations.
    """
    # Compute confusion matrix
    cm = confusion_matrix(pred_y, true_y)
    # Find the arrangement that maximizes the score
    r_ind, c_ind = linear_sum_assignment(cm, maximize=True)
    return np.sum(cm[r_ind, c_ind]) / np.sum(cm)
    
def problem8(filename='classification.npz'):
    """
    The file classification.npz contains a set of 3-dimensional data points "X" and 
    their labels "y". Use your class with n_components=4 to cluster the data.
    Plot the points with the predicted and actual labels, and compute and return
    your model's accuracy. Be sure to check for permuted labels.
    
    Returns:
        (float) - the GMM's accuracy on the dataset
    """
    gmm = GMM(n_components=4)

    with np.load(filename) as data:
        X = data['X']
        y = data['y']

    predictions = gmm.fit_predict(X)

    train0 = y == 0 #mask train0's,1's,2's,3's
    train1 = y == 1
    train2 = y == 2
    train3 = y == 3

    predict0 = predictions == 0 #mask prediction0's,1's,2's,3's
    predict1 = predictions == 1
    predict2 = predictions == 2
    predict3 = predictions == 3

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(121,projection='3d') #3d plotting
    ax.scatter(X[:,0][train0],X[:,1][train0],X[:,2][train0],color='red',alpha=0.2)
    ax.scatter(X[:,0][train1],X[:,1][train1],X[:,2][train1],color='blue',alpha=0.2)
    ax.scatter(X[:,0][train2],X[:,1][train2],X[:,2][train2],color='green',alpha=0.2)
    ax.scatter(X[:,0][train3],X[:,1][train3],X[:,2][train3],color='orange',alpha=0.2)
    ax.set_title("Train")

    ax = fig.add_subplot(122,projection='3d')
    ax.scatter(X[:,0][predict0],X[:,1][predict0],X[:,2][predict0],color='red',alpha=0.2)
    ax.scatter(X[:,0][predict1],X[:,1][predict1],X[:,2][predict1],color='blue',alpha=0.2)
    ax.scatter(X[:,0][predict2],X[:,1][predict2],X[:,2][predict2],color='green',alpha=0.2)
    ax.scatter(X[:,0][predict3],X[:,1][predict3],X[:,2][predict3],color='orange',alpha=0.2)
    ax.set_title("Predictions")

    plt.show()

    return get_accuracy(predictions,y)

# Problem 9
def problem9(filename='classification.npz'):
    """
    Again using classification.npz, compare your class, sklearn's GMM implementation, 
    and sklearn's K-means implementation for speed of training and for accuracy of 
    the resulting clusters. Print your results. Be sure to check for permuted labels.
    """
    with np.load(filename) as data:
        X = data['X']
        y = data['y']

    skgmm = mixture.GaussianMixture(4,max_iter=200) #SKLEARN GMM
    a = time.time()
    preds = skgmm.fit_predict(X)
    b = time.time()
    print(f"SKLearn GMM Accuracy={get_accuracy(preds,y)}")
    print(f"SKlearn GMM Time={b-a}")

    skkm = cluster.KMeans(4,max_iter=200,tol=1e-3) #SKLEARN KMEANS
    a = time.time()
    preds = skkm.fit_predict(X)
    b = time.time()
    print(f"SKLearn Kmeans Accuracy={get_accuracy(preds,y)}")
    print(f"SKlearn Kmeans Time={b-a}")

    gmm = GMM(n_components=4) #GMM implementation
    a = time.time()
    preds = gmm.fit_predict(X)
    b = time.time()
    print(f"My GMM Accuracy={get_accuracy(preds,y)}")
    print(f"My GMM Time={b-a}")


def test_2():
    weights = np.array([0.6, 0.4])
    means = np.array([[-0.5, -4.0], [0.5, 0.5]])
    covars = np.array([[[1, 0],[0, 1]],[[0.25, -1],[-1, 8]]])
    gmm = GMM(n_components=2,weights=weights,means=means,covars=covars)
    assert np.isclose(gmm.pdf(np.array([1.0,-3.5])),0.05077912539363083)
    assert np.isclose(gmm.component_logpdf(0,np.array([1.0,-3.5])),-3.598702690175336)
    assert np.isclose(gmm.component_logpdf(1, np.array([1.0, -3.5])),-3.7541677982835004)

def test_4():
    data = np.array([[0.5, 1.0],[1.0, 0.5],[-2.0, 0.7]])
    weights = np.array([0.6, 0.4])
    means = np.array([[-0.5, -4.0], [0.5, 0.5]])
    covars = np.array([[[1, 0],[0, 1]],[[0.25, -1],[-1, 8]]])
    gmm = GMM(n_components=2,weights=weights,means=means,covars=covars)
    e_step = gmm._compute_e_step(data)
    answer = np.array([[3.49810771e-06, 5.30334386e-05, 9.99997070e-01],[9.99996502e-01, 9.99946967e-01, 2.93011749e-06]])
    assert np.allclose(e_step,answer)

def test_5():
    data = np.array([[0.5, 1.0],[1.0, 0.5],[-2.0, 0.7]])
    weights = np.array([0.6, 0.4])
    means = np.array([[-0.5, -4.0], [0.5, 0.5]])
    covars = np.array([[[1, 0],[0, 1]],[[0.25, -1],[-1, 8]]])
    gmm = GMM(n_components=2,weights=weights,means=means,covars=covars)
    m_step = gmm._compute_m_step(data)
    answer = (np.array([0.3333512, 0.6666488]),np.array([[-1.99983216, 0.69999044],[ 0.74998978, 0.75000612]]),np.array([[[ 4.99109197e-04, -2.91933135e-05],[-2.91933135e-05, 2.43594533e-06]],[[ 6.25109881e-02, -6.24997069e-02],[-6.24997069e-02, 6.24999121e-02]]]))
    for i in range(len(answer)):
        assert np.allclose(answer[i],m_step[i])

def test_all():
    test_2()
    test_4()
    test_5()



