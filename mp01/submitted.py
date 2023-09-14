'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np


def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''

    n = len(texts)
    count0, count1 = [], []
    for text in texts:
        count0.append(text.count(word0))
        count1.append(text.count(word1))

    max0 = max(count0)
    max1 = max(count1)

    c = list(zip(count0, count1))

    Pjoint = np.ndarray((max0 + 1, max1 + 1))

    for i in range(max0 + 1):
        for j in range(max1 + 1):
            Pjoint[i, j] = c.count((i, j)) / n

    # raise RuntimeError('You need to write this part!')
    return Pjoint


def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    row, col = Pjoint.shape
    if index == 0:
        Pmarginal = np.zeros(row)
        for i in range(row):
            for j in range(col):
                Pmarginal[i] += Pjoint[i, j]
    else:
        Pmarginal = np.zeros(col)
        for i in range(col):
            for j in range(row):
                Pmarginal[i] += Pjoint[j, i]

    # raise RuntimeError('You need to write this part!')
    return Pmarginal


def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''

    m, n = Pjoint.shape
    Pcond = np.ndarray((m, n))
    for i in range(m):
        for j in range(n):
            Pcond[i, j] = Pjoint[i, j] / Pmarginal[i]

    # raise RuntimeError('You need to write this part!')
    return Pcond


def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''

    mu = 0
    for i in range(len(P)):
        mu += P[i] * i

    # raise RuntimeError('You need to write this part!')
    return mu


def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    E = mean_from_distribution(P)
    P1 = np.zeros(P.shape)
    var = 0
    for i in range(len(P)):
        var += P[i] * (i - E) ** 2
    # raise RuntimeError('You need to write this part!')
    return var


def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''

    m, n = P.shape
    Px = marginal_distribution_of_word_counts(P, 0)
    Py = marginal_distribution_of_word_counts(P, 1)
    Ex = mean_from_distribution(Px)
    Ey = mean_from_distribution(Py)
    covar = 0
    for i in range(m):
        for j in range(n):
            covar += P[i, j] * ((i - Ex) * (j - Ey))

    # raise RuntimeError('You need to write this part!')
    return covar


def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''

    m, n = P.shape
    expected=0
    for i in range(m):
        for j in range(n):
            expected+=f(i, j)*P[i, j]

    # raise RuntimeError('You need to write this part!')
    return expected
