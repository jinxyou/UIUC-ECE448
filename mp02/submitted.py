'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter
import math

stopwords = set(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren", "'t", "as",
     "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot",
     "could", "couldn", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each", "few", "for",
     "from", "further", "had", "hadn", "has", "hasn", "have", "haven", "having", "he", "he", "'d", "he", "'ll", "he",
     "'s", "her", "here", "here", "hers", "herself", "him", "himself", "his", "how", "how", "i", "'m", "'ve", "if",
     "in", "into", "is", "isn", "it", "its", "itself", "let", "'s", "me", "more", "most", "mustn", "my", "myself", "no",
     "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
     "own", "same", "shan", "she", "she", "'d", "she", "ll", "she", "should", "shouldn", "so", "some", "such", "than",
     "that", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "there", "these", "they", "they",
     "they", "they", "'re", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
     "wasn", "we", "we", "we", "we", "we", "'ve", "were", "weren", "what", "what", "when", "when", "where", "where",
     "which", "while", "who", "who", "whom", "why", "why", "with", "won", "would", "wouldn", "you", "your", "yours",
     "yourself", "yourselves"])


def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    # raise RuntimeError("You need to write this part!")

    frequency = {}
    for key in train.keys():
        frequency_class = Counter()
        for text in train[key]:
            for word in text:
                frequency_class[word] += 1
        frequency[key] = frequency_class

    return frequency


def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    # raise RuntimeError("You need to write this part!")

    nonstop = {}

    for clas in frequency.keys():
        nonstop[clas] = frequency[clas].copy()
        for word in stopwords:
            if word in nonstop[clas]:
                del nonstop[clas][word]
    return nonstop


def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    # raise RuntimeError("You need to write this part!")

    likelihood = {}
    for clas in nonstop.keys():
        likelihood[clas] = {}
        sum(nonstop[clas].values())

        for token in nonstop[clas].keys():
            likelihood[clas][token] = (nonstop[clas][token] + smoothness) / (
                    sum(nonstop[clas].values()) + smoothness * (len(nonstop[clas].keys()) + 1))
        likelihood[clas]['OOV'] = smoothness / (
                sum(nonstop[clas].values()) + smoothness * (len(nonstop[clas].keys()) + 1))

    return likelihood


def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    # raise RuntimeError("You need to write this part!")
    prior_prob = {'pos': prior, 'neg': 1 - prior}
    posterior = []
    for text in texts:
        posterior.append({})

    for clas in likelihood.keys():
        for i in range(len(texts)):
            posterior[i][clas] = math.log(prior_prob[clas])

    for clas in likelihood.keys():
        for i in range(len(texts)):
            for token in texts[i]:
                if token in stopwords:
                    continue
                if token not in likelihood[clas].keys():
                    posterior[i][clas] += math.log(likelihood[clas]['OOV'])
                    continue
                posterior[i][clas] += math.log(likelihood[clas][token])

    hypotheses = []
    for i in range(len(texts)):
        if posterior[i]['pos'] > posterior[i]['neg']:
            hypotheses.append('pos')
        elif posterior[i]['pos'] < posterior[i]['neg']:
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')

    return hypotheses


def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    # raise RuntimeError("You need to write this part!")

    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for m in range(len(priors)):
        for n in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[n])
            hypothese = naive_bayes(texts, likelihood, priors[m])
            corr = 0
            length = len(labels)
            for i in range(length):
                if hypothese[i] == labels[i]:
                    corr += 1
            accuracies[m, n] = corr / length
    return accuracies
