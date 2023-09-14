# reader.py
# ---------------
# Written by Mark Hasegawa-Johnson, February 2023.
# CC0: this file has been released into the public domain,
# as described at https://wiki.creativecommons.org/wiki/CC0.
"""
Read in reasoning environments from one of the RuleTaker metadata jsonl files
distributed by https://allenai.org/data/ruletaker.
"""
import os, json
import numpy as np

def parse_triple(s):
    '''
    Parse the representation of a logical triple.
   
    @param s (str) - a string of the form '("head" "predicate" "tail" "negation")'
    
    @return t (list) - a proposition, a list of the form [head, predicate, tail, negation]
    '''
    s = s.strip() # remove leading and trailing whitespace
    if s[0] != '(' or s[-1] != ')':
        raise RuntimeError("parse_triple input should start with '(' and end with ')': %s"%(s))
    s = s[1:-1] # remove leading and trailing parentheses
    proposition = [ word.strip() for word in s.split('"') if not len(word.strip())==0 ]
    if proposition[3] == "+":
        proposition[3]=True
    elif proposition[3]=="~" or proposition[3]=="-":
        proposition[3]=False
    else:
        raise RuntimeError("Last element of logical triple should be +, -, or ~: %s"%(s))
    return proposition

def parse_rule(s):
    '''
    Parse the representation of a rule or rule

    @param s (str) - string of the form ((ante1, ante2, ...) -> cons)
       where ante1, ... and cons are all strings representing logical triples

    @return antecedents (list) - list of antecedent propositions
    @return consequent (list) - consequent proposition
       where each proposition is a list of the form [ head, predicate, tail, negation ]
    '''
    s = s.strip() # strip leading and trailing spaces
    if s[0] != '(' or s[-1] != ')':
        raise RuntimeError("parse_rule input should start with '(' and end with ')': %s"%(s))
    s = s[1:-1] # remove leading and trailing parentheses

    ruleparts = s.split("->")
    if len(ruleparts) != 2:
        raise RuntimeError("parse_rule input should contain exactly one '->' symbol: %s"%(s))
    a = ruleparts[0].strip() # strip off trailing whitespace
    if a[0] != '(' or a[-1] != ')':
        raise RuntimeError("parse_rule antecedent list should start with '(' and end with ')': %s"%(a))
    a = a[1:-1] # remove leading and trailing parentheses from the antecedent list
    antecedents = []
    while len(a) > 0:
        n = a.find(")") # end at first close-paren
        antecedents.append(parse_triple(a[:(n+1)]))
        a = a[(n+1):]

    consequent = parse_triple(ruleparts[1])
    return antecedents, consequent
        
def load_datafile(filename):
    '''
    Load a RuleTaker jsonl file in a format suitable for forward-chaining.

    @param filename (str): the file containing the data.  Must be in jsonl format.

    @return worlds (dict): a dict mapping world IDs to worlds
      Each world is a dict containing two entries: 
      world['rules'] - a dict mapping rule IDs to rules.
        Each rule is a dict:
        rule['text'] is the natural language text of the rule
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
      world['questions'] - a dict mapping question IDs to questions.
        Each question is a dict:
        question['text'] is the natural language text of the question
        question['proofs'] is a string specifying the reference proof of the provided answer
        question['query'] is a list specifying the rule in standard proposition format
        question['answer'] indicates the correct answer, which may be one of:
           True: query is true
           False: query is false
           "NAF": supported by a negative statement which wasn't disproven
           "CWA": search for a proof exceeded maximum depth
      Standard proposition format is a list of length 4: [head, predicate, tail, negation]
        where negation is True or False.
    '''
    worlds = {}
    with open(filename, 'r') as f:
        for line in f:
            d = json.loads(line)
            id = d['id']
            # Convert triples into rules with empty antecedents
            rules = {}
            for (k,v) in d['triples'].items():
                rules[k] = {
                    'text': v['text'],
                    'antecedents': [],
                    'consequent': parse_triple(v['representation'])
                }
            for (k,v) in d['rules'].items():
                antecedents, consequent = parse_rule(v['representation'])
                rules[k] = {
                    'text': v['text'],
                    'antecedents': antecedents,
                    'consequent': consequent
                }
            # create the list of questions
            questions = {}
            for (k,v) in d['questions'].items():
                answer=v['answer']
                if "NAF" in v['proofs']:
                    answer="NAF"
                if "CWA" in v['proofs']:
                    answer="CWA"
                questions[k] = {
                    'text': v['question'][:-1]+"?",
                    'proofs': v['proofs'],
                    'answer': answer,
                    'query' : parse_triple(v['representation'])
                    }
            worlds[id]={'rules':rules, 'questions':questions}
    return worlds
