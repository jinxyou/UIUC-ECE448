'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np


def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''

    # raise RuntimeError('You need to write this part!')
    n, img = train_images.shape
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(image - train_images[i, :])
    idx = np.argpartition(distances, k)
    neighbors = np.ndarray((k, img))
    labels = np.ndarray(k, dtype=bool)
    for i in range(k):
        neighbors[i] = train_images[idx[i]]
        labels[i] = train_labels[idx[i]]
    return neighbors, labels


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''

    # raise RuntimeError('You need to write this part!')
    hypotheses, scores = [], []
    for img in dev_images:
        neighbors, labels = k_nearest_neighbors(img, train_images, train_labels, k)
        y = np.bincount(labels)
        maximum = max(y)
        for i in range(len(y)):
            if y[i] == maximum:
                if y[i] != k / 2:
                    hypotheses.append(i)
                    scores.append(y[i])
                else:
                    hypotheses.append(0)
                    scores.append(y[i])
                break
    return hypotheses, scores

def compute(hypotheses, references, h, r):
    count=0
    M=len(hypotheses)
    for i in range(M):
        if hypotheses[i]==h and references[i]==r:
            count+=1
    return count

def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    # raise RuntimeError('You need to write this part!')
    n=max(hypotheses)
    m=max(references)
    confusions=np.zeros((m+1,n+1))
    m, n = confusions.shape
    for i in range(m):
        for j in range(n):
            confusions[i, j]=compute(hypotheses, references, j, i)
    precision=confusions[1,1]/(confusions[1,1]+confusions[0,1])
    recall=confusions[1,1]/(confusions[1,1]+confusions[1,0])
    accuracy=(confusions[1,1]+confusions[0,0])/(confusions[1,1]+confusions[0,0]+confusions[0,1]+confusions[1,0])
    f1=2/(1/recall+1/precision)
    return confusions, accuracy, f1

