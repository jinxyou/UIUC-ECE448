import unittest, json, submitted
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight

# TestSequence
class TestStep(unittest.TestCase):
    @weight(8)
    def test_choose_unexplored_action(self):
        q_learner = submitted.q_learner(0.05,0.05,0.99,1,[10,10,2,2,10])
        actions = []
        for i in range(4):
            actions.append(q_learner.choose_unexplored_action([9,9,1,1,9]))
        self.assertEqual(np.count_nonzero([y==1 for y in actions[:3]]),1,
                         msg='''action==1 should occur once in the first 3 calls to act: %s
                         '''%(str(actions[:3])))
        self.assertEqual(np.count_nonzero([y==0 for y in actions[:3]]),1,
                         msg='''action==0 should occur once in the first 3 calls to act: %s
                         '''%(str(actions[:3])))
        self.assertEqual(np.count_nonzero([y==-1 for y in actions[:3]]),1,
                         msg='''action==-1 should occur once in the first 3 calls to act: %s
                         '''%(str(actions[:3])))
        self.assertEqual(actions[3],None,
                            msg='''choose_unexplored_action fourth action should be None, not %s
                            '''%(actions[3]))

    @weight(8)
    def test_q_local(self):
        q_learner = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,10,10])
        q_local=q_learner.q_local(6.25,[9,9,1,1,9])
        self.assertAlmostEqual(q_local,6.25,
                               msg='''q_local(6.25,...) should give value 6.25, not %g
                               '''%(q_local))

    @weight(8)
    def test_q_learn(self):
        q_learner = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
        q_learner.learn([9,9,1,1,9],-1,6.25,[0,0,0,0,0])
        q1 = np.amax(q_learner.report_q([9,9,1,1,9]))
        self.assertAlmostEqual(q1,0.3125,
                               msg='''q_learner.learn should give Q value of 0.3125, not %g
                               '''%(q1))
        q_learner.learn([9,9,1,1,8],1,3.1,[9,9,1,1,9])
        q2 = np.amax(q_learner.report_q([9,9,1,1,8]))
        self.assertAlmostEqual(q2,0.17046875,
                               msg='''q_learner.learn should give Q value of 0.17046875, not %g
                               '''%(q2))

    @weight(8)
    def test_exploit(self):
        q_learner1 = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
        q_learner1.learn([9,9,1,1,9],-1,6.25,[0,0,0,0,0])
        action, q = q_learner1.exploit([9,9,1,1,9])
        self.assertEqual(action, -1,
                         '''exploit should give an action of -1, not %d
                         '''%(action))
        self.assertAlmostEqual(q, 0.3125,
                               msg='''exploit should give a q-value of 0.3125, not %g
                               '''%(q))

    @weight(8)
    def test_act(self):
        q_learner=submitted.q_learner(0.05,0.25,0.99,1,[10,10,2,2,10])
        q_learner.learn([9,9,1,1,9],-1,6.25,[0,0,0,0,0])
        actions = np.zeros(15)
        for i in range(15):
            actions[i] = q_learner.act([9,9,1,1,9])
        self.assertEqual(np.count_nonzero(actions[:3]==1),1,
                         msg='''action==1 should occur once in the first 3 calls to act: %s
                         '''%(str(actions[:3])))
        self.assertEqual(np.count_nonzero(actions[:3]==0),1,
                         msg='''action==0 should occur once in the first 3 calls to act: %s
                         '''%(str(actions[:3])))
        self.assertEqual(np.count_nonzero(actions[:3]==-1),1,
                         msg='''action==-1 should occur once in the first 3 calls to act: %s
                         '''%(str(actions[:3])))
        self.assertNotEqual(np.count_nonzero(actions[3:15]==actions[3]),12,
                            msg='''The actions are all %d; they should not be
                            '''%(np.count_nonzero(actions[3:15]==actions[3])))
                         
    @weight(10)
    def test_trained_model(self):
        import pong
        state_quantization = [10,10,2,2,10]
        q_learner=submitted.q_learner(0.05,0.05,0.99,5,state_quantization)
        q_learner.load('trained_model.npz')
        pong_game = pong.PongGame(learner=q_learner, visible=False, state_quantization=state_quantization)
        scores, q_achieved, q_states = pong_game.run(m_games=10, states=[])
        self.assertGreater(np.average(scores),6,
                           msg='''trained_model.npz average score is %g which is not above 6
                           '''%(np.average(scores)))
