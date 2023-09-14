'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np


# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!")
    word_map = dict()
    general_tag_count = defaultdict(int)
    for sentence in train:
        for word, tag in sentence:
            general_tag_count[tag] += 1
            if word not in word_map.keys():
                word_map[word] = defaultdict(int)
            word_map[word][tag] += 1

    result = []
    for sentence in test:
        new_sentence = []
        for word in sentence:
            if word in word_map.keys():
                tagcount = word_map[word]
                maxtag = max(tagcount, key=tagcount.get)
                new_sentence.append((word, maxtag))
            else:
                maxtag = max(general_tag_count, key=general_tag_count.get)
                new_sentence.append((word, maxtag))
        result.append(new_sentence)

    return result


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!")

    tag_count = defaultdict(int)  # count occurrences of each tag
    tag_pair_count = defaultdict(int)  # count occurrences of each tag pair
    tag_word_count = defaultdict(int)  # count occurrences of each tag/word pair
    word_count = defaultdict(int)

    for sentence in train:
        prev_tag = "START"
        for word, tag in sentence[1::]:
            tag_count[tag] += 1
            tag_pair_count[(prev_tag, tag)] += 1
            tag_word_count[(tag, word)] += 1
            word_count[word] += 1
            prev_tag = tag

    alpha = 0.1

    initial_probs = defaultdict(float)

    # Transition Probs & Initial Probs
    tag_pair_total = defaultdict(int)
    for tag_pair in tag_pair_count:
        tag_pair_total[tag_pair[0]] += tag_pair_count[tag_pair]
    transition_probs = defaultdict(float)
    start_count = defaultdict(int)
    for tag_pair in tag_pair_count:
        transition_probs[tag_pair] = (tag_pair_count[tag_pair] + alpha) / \
                                     (tag_pair_total[tag_pair[0]] + alpha * len(tag_count))
        if tag_pair[0] == 'START':
            initial_probs[tag_pair[1]] = (tag_pair_count[tag_pair] + alpha) / \
                                         (tag_pair_total[tag_pair[0]] + alpha * len(tag_count))
            start_count[tag_pair[1]] += 1

    # Compute emission probabilities
    tag_word_total = defaultdict(int)
    for tag_word in tag_word_count:
        tag_word_total[tag_word[0]] += tag_word_count[tag_word]
    emission_probs = defaultdict(float)
    for tag_word in tag_word_count:
        emission_probs[tag_word] = (tag_word_count[tag_word] + alpha) / \
                                   (tag_word_total[tag_word[0]] + alpha * len(word_count))
        emission_probs[tag_word[0], 'UNKNOWN'] = alpha / \
                                                 (tag_word_total[tag_word[0]] + alpha * len(word_count))

    initial_probs_log = {tag: math.log(prob) for tag, prob in initial_probs.items()}
    transition_probs_log = {(tag_pair[0], tag_pair[1]): math.log(prob) for tag_pair, prob in transition_probs.items()}
    emission_probs_log = {(tag_word[0], tag_word[1]): math.log(prob) for tag_word, prob in emission_probs.items()}

    # Step 4: Construct the trellis
    paths=[]
    for sentence in test:
        trellis = [{} for _ in range(len(sentence))]
        # Initialize the start state
        for tag in tag_count:
            prob = initial_probs_log.get(tag, 0) + emission_probs_log.get((tag, sentence[0]), emission_probs_log[(tag, 'UNKNOWN')])
            trellis[0][tag] = {'prob': prob, 'prev_tag': None}
        # Fill in the rest of the trellis
        for i in range(1, len(sentence)):
            for tag in tag_count:
                probs = {}
                for prev_tag in trellis[i - 1]:
                    trans_prob = transition_probs_log.get((prev_tag, tag), float('-inf'))
                    emit_prob = emission_probs_log.get((tag, sentence[i]), emission_probs_log[(tag, 'UNKNOWN')])
                    probs[prev_tag] = trellis[i - 1][prev_tag]['prob'] + trans_prob + emit_prob
                best_prev_tag = max(probs, key=probs.get)
                trellis[i][tag] = {'prob': probs[best_prev_tag], 'prev_tag': best_prev_tag}
        # Return the best path through the trellis
        path = []
        max_prob = float('-inf')
        for tag in trellis[-1]:
            if trellis[-1][tag]['prob'] > max_prob:
                max_prob = trellis[-1][tag]['prob']
                best_tag = tag
        path.append(best_tag)
        for i in range(len(sentence) - 1, 0, -1):
            prev_tag = trellis[i][best_tag]['prev_tag']
            path.insert(0, prev_tag)
            best_tag = prev_tag

        paths.append(list(zip(sentence, path)))

    return paths


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")
