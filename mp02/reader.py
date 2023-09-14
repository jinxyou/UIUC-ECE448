# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
"""
This file is responsible for providing functions for reading the files
"""
import os
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

porter_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
bad_words = {'aed','oed','eed'} # these words fail in nltk stemmer algorithm

def loadFile(filename, stemming, lower_case):
    '''
    Load a file, and returns a list of words.

    Parameters:
    filename (str): the directory containing the data
    stemming (bool): if True, use NLTK's stemmer to remove suffixes
    lower_case (bool): if True, convert letters to lowercase

    Output:
    x (list): x[n] is the n'th word in the file
    '''
    text = []
    with open(filename, 'rb') as f:
        for line in f:
            if lower_case:
                line = line.decode(errors='ignore').lower()
                text += tokenizer.tokenize(line)
            else:
                text += tokenizer.tokenize(line.decode(errors='ignore'))
    if stemming:
        for i in range(len(text)):
            if text[i] in bad_words:
                continue
            text[i] = porter_stemmer.stem(text[i])
    return text
    
def loadDir(dirname,stemming,lower_case, use_tqdm=True):
    '''
    Loads the files in the folder and returns a 
    list of lists of words from the text in each file.

    Parameters:
    dirname (str): the directory containing the data
    stemming (bool): if True, use NLTK's stemmer to remove suffixes
    lower_case (bool): if True, convert letters to lowercase
    use_tqdm (bool, default:True): if True, use tqdm to show status bar

    Output:
    texts (list of lists): texts[m][n] is the n'th word in the m'th email
    count (int): number of files loaded
    '''
    texts = []
    count = 0
    if use_tqdm:
        for f in tqdm(sorted(os.listdir(dirname))):
            texts.append(loadFile(os.path.join(dirname,f), stemming, lower_case))
            count = count+1
    else:
        for f in sorted(os.listdir(dirname)):
            texts.append(loadFile(os.path.join(dirname,f), stemming, lower_case))
            count = count+1
    return texts, count

def loadTrain(dirname,stemming,lower_case, use_tqdm=True):
    '''
    Loads a training dataset.

    Parameters:
    dirname (str): the directory containing the data
        - dirname/y should contain training examples from class y

    stemming (bool): if True, use NLTK's stemmer to remove suffixes
    lower_case (bool): if True, convert letters to lowercase
    use_tqdm (bool, default:True): if True, use tqdm to show status bar

    Output:
    train (dict of list of lists): 
        - train[y][i][k] = k'th token of i'th text of class y
    '''
    train = {}
    for y in sorted(os.listdir(dirname)):
        train[y], y_count = loadDir(os.path.join(dirname,y), stemming, lower_case, use_tqdm)
    return train
    
def loadDev(dirname,stemming,lower_case, use_tqdm=True):
    '''
    Loads a dev dataset.

    Parameters:
    dirname (str): the directory containing the data
        - dirname/y should contain training examples from class y

    stemming (bool): if True, use NLTK's stemmer to remove suffixes
    lower_case (bool): if True, convert letters to lowercase
    use_tqdm (bool, default:True): if True, use tqdm to show status bar

    Output:
    texts (list of lists): 
        - texts[i][k] = k'th token of i'th text
    labels (list):
        - labels[i] = label of the i'th text
    '''
    texts = []
    labels = []
    for y in sorted(os.listdir(dirname)):
        newtexts, count = loadDir(os.path.join(dirname,y), stemming, lower_case, use_tqdm)
        labels.extend([y]*len(newtexts))
        texts.extend(newtexts)
    return texts, labels


    
