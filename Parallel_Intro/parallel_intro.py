# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
import numpy as np
import time
from matplotlib import pyplot as plt

# Problem 1
def initialize(blocking=True,closing=True):
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    client = Client() #initialize client
    dview = client[:] #visualize the engines
    dview.execute("import scipy.sparse as sparse")
    dview.block = blocking #initialize the blocking feature
    if closing:
        client.close() #close the client
        return dview #return the view
    else:
        return dview, client

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    dview, client = initialize(blocking=True,closing=False) #initialize dview and client with scipy (who cares it can't hurt)
    dview.push(dx) #push the variable dictionary
    for key in dx.keys():
        assert dx[key] == dview.pull(key)[0] #peform the grand verification
    client.close()

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    dview, client = initialize(blocking=True,closing=False)
    dview['n'] = n
    dview.execute("import numpy as np")
    dview.execute("x = np.random.normal(size=n)") #standard normal distribution
    dview.execute("mu = np.mean(x)")
    dview.execute("min = np.min(x)")
    dview.execute("max = np.max(x)")
    means = dview.pull("mu") #save before closing
    mins = dview.pull("min")
    maxs = dview.pull("max")
    client.close()

    return means,mins,maxs


# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    ns = [1000000,5000000,10000000,15000000]
    parallel_times, serial_times = [], []

    N = len(initialize().targets)

    for n in ns:
        a_p = time.time() #time it in parallel
        prob3(n)
        parallel_times.append(time.time()-a_p)

        a_s = time.time()
        for _ in range(N): #time it in nonparallel
            data = np.random.normal(size=n)
            np.mean(data)
            np.min(data)
            np.max(data)
        serial_times.append(time.time()-a_s)

    plt.plot(ns,parallel_times,label="parallel")
    plt.plot(ns,serial_times,label="serial")
    plt.legend()
    plt.xlabel("size")
    plt.ylabel("run time")
    plt.show()

# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    dview, client = initialize(blocking=True,closing=False) 
    grid = np.linspace(a,b,n)
    h = abs(grid[1] - grid[0]) #space between grid points
    def traprule(x1,x2): #map this
        return f(x1) + f(x2)
    dview.scatter("grid",grid)
    dview['func'] = f
    dview['h'] = h
    parallel_computed = dview.map(traprule,grid[:-1],grid[1:])
    client.close() #close the client
    return np.sum(parallel_computed)*(h/2)

def testprob5():
    f = lambda x: x
    a = 0
    b = 1
    print(parallel_trapezoidal_rule(f,a,b,n=1000))
