import unittest, json, reader, submitted, copy
from gradescope_utils.autograder_utils.decorators import weight

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        self.filename = 'data/sample_data.jsonl'
        self.worldname = 'RelNoneg-D2-2133'

    @weight(12.5)
    def test_standardization(self):
        worlds = reader.load_datafile(self.filename)
        world = worlds[self.worldname]
        standardized, variables = submitted.standardize_variables(world['rules'])
        vars = list(variables)
        usedvars = []
        for (ruleid,rule) in world['rules'].items():
            self.assertIn(ruleid,standardized,
                          'Output of standardize_variables missing rule %s'%(ruleid))
            self.assertIn('antecedents',standardized[ruleid],
                          'Output of standardize_variables rule %s missing antecedents'%(ruleid))
            self.assertIn('consequent',standardized[ruleid],
                          'Output of standardize_variables rule %s missing consequent'%(ruleid))
            ref = [ rule['consequent'] ] + list(rule['antecedents'])
            hyp = [ standardized[ruleid]['consequent'] ] + list(standardized[ruleid]['antecedents'])
            self.assertEqual(len(ref),len(hyp),
                             '''standardized_rules[%s] should have %d antecedents, 
                             not %d'''%(ruleid,len(ref)-1,len(hyp)-1))
            myvar = None
            for r,h in zip(ref,hyp):
                self.assertEqual(len(h),4,
                                 '''standardize_variables output facts should all be 4-tuples: 
                                 %s'''%(str(h)))
                for i in range(3):
                    if r[i]=="something":
                        if myvar==None:
                            myvar=h[i]
                            self.assertIn(myvar,vars,
                                          '''standardize_variables return "standardized" contains %s,
                                          but "variables" does not: %s'''%(myvar,str(h)))
                            vars.remove(myvar)
                            self.assertNotIn(myvar, usedvars,
                                             '''standardize_variables rule %s contains variable %s
                                             that was also used by a different rule: 
                                             %s'''%(ruleid,myvar,str(standardized[ruleid])))
                            usedvars.append(myvar)
                        else:
                            self.assertEqual(h[i], myvar,
                                             '''standardize_variables rule includes two
                                             different variables, 
                                             %s and %s: %s'''%(h[i],myvar,str(standardized[ruleid])))
        self.assertEqual(len(vars),0,
                         '''standardize_variables output "variables" contains variables
                            that were not used in any rule: %s 
                            (%s)'''%(str(vars),str(standardized)))
            
    @weight(12.5)
    def test_unify(self):
        unification, subs = submitted.unify(['x','eats','y',False],['a','eats','b',False],['x','y','a','b'])
        tunify = ['a','eats','b',False]
        tsubs = {'x':'a', 'y':'b'}
        self.assertEqual(tuple(tunify), tuple(unification),
                         '''unification should be %s, not %s'''%(tunify,unification))
        self.assertEqual(tuple(sorted(tsubs.items())), tuple(sorted(subs.items())),
                         '''subs should be %s, not %s'''%(tsubs,subs))
        unification, subs = submitted.unify(['bobcat','eats','y',True],['a','eats','squirrel',True],['x','y','a','b'])
        tunify = ['bobcat', 'eats', 'squirrel', True]
        tsubs = {'a': 'bobcat', 'y': 'squirrel'}
        self.assertEqual(tuple(tunify), tuple(unification),
                         '''unification should be %s, not %s'''%(tunify,unification))
        self.assertEqual(tuple(sorted(tsubs.items())), tuple(sorted(subs.items())),
                         '''subs should be %s, not %s'''%(tsubs,subs))
        unification, subs = submitted.unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
        tunify = ['bobcat', 'eats', 'bobcat', True]
        tsubs = {'x': 'a', 'a': 'bobcat'}
        self.assertEqual(tuple(tunify), tuple(unification),
                         '''unification should be %s, not %s'''%(tunify,unification))
        self.assertEqual(tuple(sorted(tsubs.items())), tuple(sorted(subs.items())),
                         '''subs should be %s, not %s'''%(tsubs,subs))
        unification, subs = submitted.unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
        tunify = ['bobcat', 'eats', 'bobcat', True]
        tsubs= {'a': 'x', 'x': 'bobcat'}
        self.assertEqual(tuple(tunify), tuple(unification),
                         '''unification should be %s, not %s'''%(tunify,unification))
        self.assertEqual(tuple(sorted(tsubs.items())), tuple(sorted(subs.items())),
                         '''subs should be %s, not %s'''%(tsubs,subs))
        unification, subs = submitted.unify(['a','eats','bobcat',True],['x','eats','x',False],['x','y','a','b'])
        self.assertIsNone(unification,
                        '''unification should be None, not %s'''%(unification))
        self.assertIsNone(subs,
                        '''subs should be None, not %s'''%(subs))

    @weight(12.5)
    def test_apply(self):
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
        applications, goalsets = submitted.apply(rule, goals, variables)
        tapps = [{'antecedents': [['bobcat', 'is', 'nice', True],
                                  ['bobcat', 'is', 'hungry', False]],
                  'consequent': ['bobcat', 'eats', 'squirrel', False]},
                 {'antecedents': [['bald eagle', 'is', 'nice', True],
                                  ['bald eagle', 'is', 'hungry', False]],
                  'consequent': ['bald eagle', 'eats', 'squirrel', False]}]
        tgs = [[['bobcat', 'visits', 'squirrel', True],
                ['bald eagle', 'eats', 'squirrel', False],
                ['bobcat', 'is', 'nice', True],
                ['bobcat', 'is', 'hungry', False]],
               [['bobcat', 'eats', 'squirrel', False],
                ['bobcat', 'visits', 'squirrel', True],
                ['bald eagle', 'is', 'nice', True],
                ['bald eagle', 'is', 'hungry', False]]]
        self.assertEqual(len(applications),2,
                         '''applications should have length %d, not %d'''%(len(tapps),len(applications)))
        self.assertEqual(len(goalsets),2,
                         '''goalsets should have length %d, not %d'''%(len(tgs),len(goalsets)))
        self.assertIn('antecedents',applications[0],
                      '''applications[0]['antecedents'] should exist but does not''')
        self.assertIn('consequent',applications[0],
                      '''applications[0]['consequent'] should exist but does not''')
        self.assertEqual(len(applications[0]['antecedents']),2,
                         '''applications[0]['antecedents'] should have length 2 but does not''')
        self.assertIn('antecedents',applications[1],
                      '''applications[1]['antecedents'] should exist but does not''')
        self.assertIn('consequent',applications[1],
                      '''applications[1]['consequent'] should exist but does not''')
        self.assertEqual(len(applications[1]['antecedents']),2,
                         '''applications[1]['antecedents'] should have length 2 but does not''')
        asets = [ tuple(sorted(tuple(x) for x in applications[i]['antecedents'])) for i in range(2) ]
        self.assertIn(tuple(sorted(tuple(x) for x in tapps[0]['antecedents'])),asets,
                      '''applications should contain a rule with antecedents %s, 
                      but does not'''%(tapps[0]['antecedents']))
        self.assertIn(tuple(sorted(tuple(x) for x in tapps[1]['antecedents'])),asets,
                      '''applications should contain a rule with antecedents %s, 
                      but does not'''%(tapps[1]['antecedents']))
        csets = [ tuple(applications[i]['consequent']) for i in range(2) ]
        self.assertIn(tuple(tapps[0]['consequent']),csets,
                      '''applications should contain a rule with consequent %s, 
                      but does not'''%(tapps[0]['consequent']))
        self.assertIn(tuple(tapps[1]['consequent']),csets,
                      '''applications should contain a rule with consequent %s, 
                      but does not'''%(tapps[1]['antecedents']))
        gsets = [ tuple(x) for gs in goalsets for x in gs ]
        for goalset in tgs:
            for goal in goalset:
                self.assertIn(tuple(goal), gsets,
                              '''one of the goalsets should include the goal %s, but it's not there'''%(goal))
        

    @weight(12.5)
    def test_backward_chain(self):
        data_worlds = reader.load_datafile(self.filename)
        world = copy.deepcopy(data_worlds[self.worldname])
        rules, variables = submitted.standardize_variables(world['rules'])
        world['rules'] = rules
        world['variables'] = variables
        for (qid, question) in world['questions'].items():
            proof = submitted.backward_chain(question['query'],world['rules'],world['variables'])
            if question['answer']==True:
                self.assertIsNotNone(proof,
                                     '''question "%s" should have non-None proof but does not'''%(question['text']))
            elif question['answer']==False:
                self.assertIsNone(proof,
                                  '''question "%s" should have proof==None but does not'''%(question['text']))
