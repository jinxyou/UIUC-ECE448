import unittest, json, reader, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        with open('solution.json') as f:
            self.solution = json.load(f)
            
    @weight(9)
    def test_joint(self):
        ref = np.array(self.solution['Pjoint'])
        texts, count = reader.loadDir('data',False,False,False)
        hyp = submitted.joint_distribution_of_word_counts(texts, 'mr', 'computer')
        (M, N) = ref.shape
        self.assertEqual(len(hyp.shape), 2,
                         'joint_distribution_of_word_counts should return a 2-dimensional array')
        self.assertLessEqual(M, hyp.shape[0],
                             'joint_distribution_of_word_counts dimension 0 should be at least %d'%(M))
        self.assertLessEqual(N, hyp.shape[1],
                             'joint_distribution_of_word_counts dimension 1 should be at least %d'%(N))
        for m in range(M):
            for n in range(N):
                self.assertAlmostEqual(ref[m,n], hyp[m,n], places=2,
                                       msg='''
                                       joint_distribution_of_word_counts[%d,%d] should be %g, not %g
                                       '''%(m,n,ref[m,n],ref[m,n]))

    @weight(9)
    def test_marginal(self):
        ref = np.array(self.solution['P1'])
        Pjoint = np.array(self.solution['Pjoint'])
        hyp = submitted.marginal_distribution_of_word_counts(Pjoint, 1)
        N = len(ref)
        self.assertLessEqual(N, len(hyp),
                             '''
                             marginal_distribution_of_word_counts(Pjoint, 1) should have length at
                             least %d, instead it is %d
                             '''%(N,len(hyp)))
        for n in range(N):
            self.assertAlmostEqual(ref[n], hyp[n], places=2,
                                   msg='''
                                   marginal_distribution_of_word_counts(Pjoint,1)[%d] should be %g, not %g
                                   '''%(n,ref[n],ref[n]))

    @weight(8)
    def test_cond(self):
        ref = np.array(self.solution['Pcond'])
        Pjoint = np.array(self.solution['Pjoint'])
        P0 = np.array(self.solution['P0'])
        hyp = submitted.conditional_distribution_of_word_counts(Pjoint, P0)
        (M,N) = ref.shape
        self.assertLessEqual(M, hyp.shape[0],
                             msg='''
                             conditional_distribution_of_word_counts dimension 0 should be %d, not %d
                             '''%(M,hyp.shape[0]))
        self.assertLessEqual(N, hyp.shape[1],
                             '''
                             conditional_distribution_of_word_counts dimension 0 should be %d, not %d
                             '''%(N,hyp.shape[1]))
        for m in range(M):
            for n in range(N):
                if not np.isnan(ref[m,n]):
                    self.assertAlmostEqual(ref[m,n], hyp[m,n], places=2,
                                           msg='''
                                           conditional_distribution_of_word_counts[%d,%d] 
                                           should be %g, not %g
                                           '''%(m,n,ref[m,n],ref[m,n]))
                
    @weight(8)
    def test_mean(self):
        ref = self.solution['mu_the']
        Pthe = np.array(self.solution['Pthe'])
        hyp = submitted.mean_from_distribution(Pthe)
        self.assertAlmostEqual(ref, hyp, places=2,
                               msg='''mean_from_distribution should be %g, not %g'''%(ref, hyp))
        
                
    @weight(8)
    def test_mean(self):
        ref = self.solution['var_the']
        Pthe = np.array(self.solution['Pthe'])
        hyp = submitted.variance_from_distribution(Pthe)
        self.assertAlmostEqual(ref, hyp, places=2,
                               msg='''variance_from_distribution should be %g, not %g'''%(ref, hyp))
        

    @weight(8)
    def test_covariance(self):
        ref = self.solution['covar_a_the']
        Pathe = np.array(self.solution['Pathe'])
        hyp = submitted.covariance_from_distribution(Pathe)
        self.assertAlmostEqual(ref, hyp, places=2,
                               msg='''covariance_from_distribution should be %g, not %g'''%(ref, hyp))
        

    @weight(8)
    def test_expected(self):
        ref = self.solution['expected']
        Pathe = np.array(self.solution['Pathe'])
        def f(x0,x1):
            return(np.log(x0+1) + np.log(x1+1))
        hyp = submitted.expectation_of_a_function(Pathe, f)
        self.assertAlmostEqual(ref, hyp, places=2,
                               msg='''expectation_of_a_function should be %g, not %g'''%(ref, hyp))
        
