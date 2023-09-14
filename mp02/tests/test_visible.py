import unittest, json, reader, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        with open('solution.json') as f:
            self.solution = json.load(f)
        self.traindir = 'data/train'
        self.devdir = 'data/dev'

    @weight(10)
    def test_frequency(self):
        ref = self.solution['frequency']
        train = reader.loadTrain(self.traindir, False, True, False)        
        hyp = submitted.create_frequency_table(train)
        self.assertTrue(hyp, 'frequency should be a dict of nonzero length')
        for y in ref.keys():
            self.assertIn(y, hyp, 'frequency should contain the member %s'%(y))
            for x in ref[y].keys():
                self.assertIn(x, hyp[y], 'frequency[%s] should contain the member %s'%(y,x))
                self.assertEqual(ref[y][x], hyp[y][x],
                                 '''
                                 frequency[%s][%s] should be %d, not %d
                                 '''%(y,x,ref[y][x], hyp[y][x],))
                
    @weight(10)
    def test_nonstop(self):
        ref = self.solution['nonstop']
        hyp = submitted.remove_stopwords(self.solution['frequency'])
        self.assertTrue(hyp, 'nonstop should be a dict of nonzero length')
        for y in ref.keys():
            self.assertIn(y, hyp, 'nonstop should contain the member %s'%(y))
            for x in ref[y].keys():
                self.assertIn(x, hyp[y], 'nonstop[%s] should contain the member %s'%(y,x))
                self.assertEqual(ref[y][x], hyp[y][x],
                                 '''
                                 nonstop[%s][%s] should be %d, not %d
                                 '''%(y,x,ref[y][x], hyp[y][x],))
                

    @weight(10)
    def test_likelihood(self):
        ref = self.solution['likelihood']
        hyp = submitted.laplace_smoothing(self.solution['nonstop'], self.solution['smoothness'])
        self.assertTrue(hyp, 'likelihood should be a dict of nonzero length')
        for y in ref.keys():
            self.assertIn(y, hyp, 'likelihood should contain the member %s'%(y))
            for x in ref[y].keys():
                self.assertIn(x, hyp[y], 'likelihood[%s] should contain the member %s'%(y,x))
                self.assertAlmostEqual(ref[y][x], hyp[y][x], places=4,
                                       msg='''
                                       likelihood[%s][%s] should be %g, not %g
                                       '''%(y,x,ref[y][x], hyp[y][x],))
                
    @weight(10)
    def test_hypotheses(self):
        ref = self.solution['hypotheses']
        texts, labels = reader.loadDev(self.devdir, False, True, False)
        hyp = submitted.naive_bayes(texts, self.solution['likelihood'], self.solution['prior'])
        self.assertTrue(hyp, 'hypotheses should be a list of nonzero length')
        self.assertEqual(len(hyp), len(ref),
                         'hypotheses should have length %d not %d'%(len(ref),len(hyp)))
        for i in range(len(ref)):
            self.assertEqual(ref[i], hyp[i],
                             msg='''
                             hypotheses[%d] should be %s, not %s
                             '''%(i,ref[i], hyp[i]))
                
    @weight(10)
    def test_accuracies(self):
        ref = np.array(self.solution['accuracies'])
        texts, labels = reader.loadDev(self.devdir, False, True, False)
        hyp = submitted.optimize_hyperparameters(texts,labels,self.solution['nonstop'],
                                                 self.solution['priors'],
                                                 self.solution['smoothnesses'])
        self.assertTrue(hyp, 'hypotheses should be a numpy array of nonzero size')
        self.assertEqual(type(hyp), np.ndarray,'accuracies should be a numpy array')
        self.assertEqual(len(hyp.shape), 2, 'accuracies should be a 2-dimensional numpy array')
        self.assertEqual(hyp.shape[0], ref.shape[0],
                         'accuracies should have %d rows, not %d'%(hyp.shape[0], ref.shape[0]))
        self.assertEqual(hyp.shape[1], ref.shape[1],
                         'accuracies should have %d columns, not %d'%(hyp.shape[0], ref.shape[0]))
        for m in range(ref.shape[0]):
            for n in range(ref.shape[1]):
                self.assertAlmostEqual(ref[m,n], hyp[m,n], places=1,
                                       msg='''
                                       accuracies[%d,%d] should be %g, not %g
                                       '''%(m, n, ref[m,n], hyp[m,n]))
                
