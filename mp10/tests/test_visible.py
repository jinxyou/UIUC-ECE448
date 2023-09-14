import unittest, json, utils, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def _test_PU(self, model_name, PU=[True, False]):
        model_file = 'models/model_%s.json'%model_name
        solution_file = 'solution_%s.json'%model_name
        model = utils.load_MDP(model_file)

        with open(solution_file, 'r') as f:
            data = json.load(f)
        P_gt = np.array(data['transition'])
        U_gt = np.array(data['utility'])

        if PU[0]:
            P = submitted.compute_transition_matrix(model)
            diff = np.abs(P - P_gt)
            expr = diff.max() < 1e-2
            subtest_name = 'Transition matrix'
            msg = 'Testing %s (%s): '%(model_file, subtest_name)
            ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
            msg += 'The difference between your transition matrix and the ground truth shoud be less than 0.01. However, your P[%d, %d, %d, %d, %d] = %.3f, while the ground truth P_gt[%d, %d, %d, %d, %d] = %.3f'%(ind[0], ind[1], ind[2], ind[3], ind[4], P[ind], ind[0], ind[1], ind[2], ind[3], ind[4], P_gt[ind])
            self.assertTrue(expr, msg)

        if PU[1]:
            U = submitted.value_iteration(model)
            diff = np.abs(U - U_gt)
            expr = diff.max() < 1e-2
            subtest_name = 'Utility function'
            msg = 'Testing %s (%s): '%(model_file, subtest_name)
            ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
            msg += 'The difference between your utility and the ground truth shoud be less than 0.01. However, your U[%d, %d] = %.3f, while the ground truth U_gt[%d, %d] = %.3f'%(ind[0], ind[1], U[ind], ind[0], ind[1], U_gt[ind])
            self.assertTrue(expr, msg)

    @weight(15)
    def test_small_P(self):
        self._test_PU('small', [True, False])

    @weight(15)
    def test_small_U(self):
        self._test_PU('small', [False, True])

    @weight(15)
    def test_large_P(self):
        self._test_PU('large', [True, False])

    @weight(15)
    def test_large_U(self):
        self._test_PU('large', [False, True])
