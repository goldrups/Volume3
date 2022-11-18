# advanced_numpy.py
"""Python Essentials: Advanced NumPy.
Samuel Goldrup
Math
27 August 2022
"""
from decimal import ROUND_DOWN
import numpy as np
from sympy import isprime
from matplotlib import pyplot as plt
from time import time

def prob1(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    B = A.copy() #make a copy so as to save our work!
    mask = B < 0
    B[mask] = 0
    return B

def prob2(arr_list):
    """return all arrays in arr_list as one 3-dimensional array
    where the arrays are padded with zeros appropriately."""

    squeezed_arrs = [np.squeeze(arr) for arr in arr_list] #squeeze them, assume 2D

    max_vert, max_hori = np.maximum.reduce([arr.shape for arr in squeezed_arrs]) #get max dims

    padded_arrs = []

    for arr in squeezed_arrs:
        vert,hori = arr.shape
        if hori < max_hori:
            x = max_hori - hori
            arr = np.hstack((arr,np.zeros((vert,x)))) #pad up!
            hori = arr.shape[1]
        if vert < max_vert:
            y = max_vert - vert
            arr = np.vstack((arr,np.zeros((y,hori)))) #pad up!
        padded_arrs.append(arr)

    return np.dstack(padded_arrs)
        
def prob3(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    return A / A.sum(axis=1).reshape(-1,1)
    
# this is provided for problem 4    
def LargestPrime(x,show_factorization=False):
    # account for edge cases.
    if x == 0 or x == 1:
        return np.nan
    
    # create needed variables
    forced_break = False
    prime_factors = [] # place to store factors of number
    factor_test_arr = np.arange(1,11)
    
    while True:
        # a factor is never more than half the number
        if np.min(factor_test_arr) > (x//2)+1:
            forced_break=True
            break
        if isprime(x):  # if the checked number is prime itself, stop
            prime_factors.append(x)
            break
        
        # check if anythin gin the factor_test_arr are factors
        div_arr = x/factor_test_arr
        factor_mask = div_arr-div_arr.astype(int) == 0
        divisors = factor_test_arr[factor_mask]
        if divisors.size > 0: # if divisors exist...
            if divisors[0] == 1 and divisors.size > 1:   # make sure not to select 1
                i = 1 
            elif divisors[0] == 1 and divisors.size == 1:  # if one is the only one don't pick it
                factor_test_arr=factor_test_arr+10
                continue
            else:   # othewise take the smallest divisor
                i = 0
            
            # if divisor was found divide number by it and 
            # repeat the process
            x = int(x/divisors[i])
            prime_factors.append(divisors[i])
            factor_test_arr = np.arange(1,11)
        else:  # if no number was found increase the test_arr 
               # and keep looking for factors
            factor_test_arr=factor_test_arr+10
            continue
    
    if show_factorization: # show entire factorization if desired
        print(prime_factors)
    if forced_break:  # if too many iterations break
        print(f"Something wrong, exceeded iteration threshold for value: {x}")
        return 0
    return max(prime_factors)

def prob4(arr,naive=False):
    """Return an array where every number is replaced be the largest prime
    in its factorization. Implement two methods. Switching between the two
    is determined by a bool.
    
    Example:
        >>> A = np.array([15, 41, 49, 1077])
        >>> prob4(A)
        array([5,41,7,359])
    """
    if naive:
        for i,num in enumerate(arr):
            arr[i] = LargestPrime(arr[i])
        return arr

    else:
        LargestPrimeVectorized = np.vectorize(LargestPrime) #vectorize the function
        arr = LargestPrimeVectorized(arr)
        return arr.astype("int32") 


def prob5(x,y,z,A,optimize=False,split=True):
    """takes three vectors and a matrix and performs 
    (np.outer(x,y)*z.reshape(-1,1))@A on them using einsum."""
    if optimize:
        return np.einsum('i,j,i,jk -> ik',x,y,z,A,optimize=optimize)
    else:
        D = np.einsum('i,j->ij',x,y)
        B = np.einsum('ij,i->ij',D,z) #break into steps if not optimizing
        C = np.einsum('ij,jk->ik',B,A)
        return C

def naive5(x,y,z,A):
    """uses normal numpy functions to do what prob5 does"""
    return np.outer(x,y)*z.reshape(-1,1)@A

def prob6():
    """Times and creates plots that generate the difference in
    speeds between einsum and normal numpy functions
    """
    sizes = list(range(3,501))
    np_times = []
    einsum_opt_times = []
    einsum_times = []
    for n in sizes: #time each method for all the sizes
        x,y,z,A = np.random.random(n),np.random.random(n),np.random.random(n),np.random.random((n,n))
        start1 = time()
        a = naive5(x,y,z,A)
        np_times.append(time() - start1)
        start2 = time()
        b = prob5(x,y,z,A,optimize=True)
        einsum_opt_times.append(time() - start2)
        assert np.allclose(a,b) == True
        start3 = time()
        c = prob5(x,y,z,A,optimize=False)
        einsum_times.append(time() - start3)
        assert np.allclose(a,c) == True

    plt.subplot(121).plot(sizes,np_times,label="Numpy") #plot the arrays
    plt.subplot(121).plot(sizes,einsum_opt_times,label="Einsum")
    plt.legend()
    plt.ylabel("Time")
    plt.xlabel("Input size")
    plt.title("Einsum Opt v. Numpy")
    plt.subplot(122).plot(sizes,np_times,label="Numpy")
    plt.subplot(122).plot(sizes,einsum_times,label="Einsum")
    plt.legend()
    plt.ylabel("Time")
    plt.xlabel("Input size")
    plt.title("Einsum v. Numpy")
    plt.tight_layout()
    plt.show()

def np_cosine(X):
    return np.cos(X)
   
def naive_cosine(X):
    for i in range(len(X)):
        X[i] = np.cos(X[i])
    return X

def better_cosine(X): #does not edit the input
    A = np.zeros(len(X))
    for i in range(len(X)):
        A[i] = np.cos(X[i])
    return A

def max_row(X):
    row_maxs = []
    for row in X:
        row_maxs.append(max(row))
    return row_maxs

n = 10**6
A = np.random.randn(n)

def prob7():
    print(A[:5])
    start1 = time()
    np_cosine(A)
    time_np = time() - start1
    print(A[:5])
    start2 = time()
    naive_cosine(A)
    time_naive = time() - start2
    print(A[:5])
    print("What happened: naive_cosine edited the array in place, but np_cosine just returned a view of the edited array")
    start3 = time()
    better_cosine(A)
    time_better = time() - start3
    print(time_np,time_naive,time_better)
    print("for loops are so much slower than broadcasting")
    B = A.reshape(10**3,10**3)
    start4 = time()
    max_row(B) #naive way
    time_max_row = time() - start4
    start5 = time()
    np.max(B,axis=1) #numpy way
    time_np_max_row = time() - start5
    print(time_max_row,time_np_max_row)
    print("Numpy method is so much faster")
    assert np.allclose(np.array(max_row(B)),np.max(B,axis=1)) == True
    print("I will never again use a for-loop if a NumPy function can be used instead.")