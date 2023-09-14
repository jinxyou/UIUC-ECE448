'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue


def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    # raise RuntimeError("You need to write this part!")

    standardized_rules = copy.deepcopy(nonstandard_rules)
    variables = []
    count = 0
    for key in standardized_rules.keys():
        isVariable = False
        antecedents = standardized_rules[key]['antecedents']
        consequent = standardized_rules[key]['consequent']
        variable = 'x' + str(count)
        for proposition in antecedents:
            for index, word in enumerate(proposition):
                if word == 'something':
                    proposition[index] = variable
                    isVariable = True
        for index, word in enumerate(consequent):
            if word == 'something':
                consequent[index] = variable
                isVariable = True
        if isVariable is True:
            variables.append(variable)
            count += 1

    return standardized_rules, variables


def update(lis, dic):
    new_lis=copy.deepcopy(lis)
    flag = True
    while flag:
        for i, e in enumerate(new_lis):
            if e in dic.keys():
                new_lis[i] = dic[e]
                break
        else:
            flag = False

    return new_lis


def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''

    # raise RuntimeError("You need to write this part!")

    var1, predicate1, var2, bool1 = query[0], query[1], query[2], query[3]
    var3, predicate2, var4, bool2 = datum[0], datum[1], datum[2], datum[3]

    if bool1 ^ bool2:
        # print(bool1, "^", bool2)
        return None, None

    if predicate1 != predicate2:
        # print("predicate")
        return None, None

    subs = dict()

    if var1 in variables:
        if var3 in variables:
            subs[var1] = var3
            var1 = var3
            var1, var2 = tuple(update([var1, var2], subs))
        else:
            subs[var1] = var3
            var1 = var3
            var1, var2 = tuple(update([var1, var2], subs))
    else:
        if var3 in variables:
            subs[var3] = var1
            var1, var2 = tuple(update([var1, var2], subs))
        else:
            if var1 != var3:
                return None, None

    if var2 in variables:
        if var4 in variables:
            subs[var2] = var4
            var2 = var4
            var1, var2 = tuple(update([var1, var2], subs))
        else:
            subs[var2] = var4
            var2 = var4
            var1, var2 = tuple(update([var1, var2], subs))
    else:
        if var4 in variables:
            subs[var4] = var2
            var1, var2 = tuple(update([var1, var2], subs))
        else:
            if var2 != var4:
                return None, None

    unification = [var1, predicate1, var2, bool1]

    return unification, subs


def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    # raise RuntimeError("You need to write this part!")

    antecedents = rule['antecedents']
    # print("antecedents:", antecedents)
    consequent = rule['consequent']
    applications = []
    goalsets = []
    for index, goal in enumerate(goals):
        # print(consequent, goal)
        unification, subs = unify(consequent, goal, variables)
        # print("unification:",unification)
        # print("subs:", subs)
        if unification is not None:
            new_consequent = unification
            new_antecedents = []
            application = dict()
            # print("antecedents:", antecedents)
            for antecedent in antecedents:
                new_antecedent = update(antecedent, subs)
                new_antecedents.append(new_antecedent)
            # print("new_antecedents:", new_antecedents)
            application['antecedents'] = new_antecedents
            application['consequent'] = new_consequent
            applications.append(application)

            new_goals = copy.deepcopy(goals)
            new_goals.pop(index)
            new_goals.extend(new_antecedents)
            goalsets.append(new_goals)

    return applications, goalsets


def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''

    # raise RuntimeError("You need to write this part!")

    def reconstruct_path(cameFrom, curr):
        path = [curr]
        # print("path:", path)
        while curr in cameFrom.keys():
            curr = list(cameFrom[curr])
            path.insert(0, curr)
        return path

    starting_state = [[query]]
    proofs = [[[]]]
    Q1 = queue.Queue()
    Q2 = queue.Queue()
    explored = ()
    explored += (starting_state,)
    Q1.put(starting_state)
    Q2.put([])
    while Q1.empty() is False:
        cur_goalset = Q1.get()[0]
        cur_applications = Q2.get()
        # print("cur_goalset:", cur_goalset)
        # print("cur_application:", cur_applications)
        if not cur_goalset:
            for proof in proofs:
                if proof[-1] == cur_applications:
                    return proof
        for rule in rules.values():
            # print(rule)
            applications, goalsets = apply(rule, cur_goalset, variables)
            if applications:
                new_state = tuple(goalsets)
                # print("new_state:", new_state)
                if new_state not in explored:
                    explored += (new_state,)
                    for proof in proofs:
                        if proof[-1] == cur_applications:
                            proofs.append(proof+ [applications])
                            # print("proof:", proof)
                    Q1.put(new_state)
                    Q2.put(applications)
    return None
