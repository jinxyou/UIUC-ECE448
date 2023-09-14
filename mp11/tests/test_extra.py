import unittest, json, submitted
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight

# TestSequence
class TestStep(unittest.TestCase):
    @weight(2)
    def test_deep_q_exceeds_8(self):
        import pong
        q_learner=submitted.deep_q(0.05,0.05,0.99,5)
        q_learner.load('trained_model.pkl')
        pong_game = pong.PongGame(learner=q_learner, visible=False)
        scores = pong_game.run(m_games=10, states=[])
        self.assertGreater(np.average(scores),8,
                           msg='''trained_model.pkl average score is %g which is not above 8
                           '''%(np.average(scores)))

    @weight(2)
    def test_deep_q_exceeds_12(self):
        import pong
        q_learner=submitted.deep_q(0.05,0.05,0.99,5)
        q_learner.load('trained_model.pkl')
        pong_game = pong.PongGame(learner=q_learner, visible=False)
        scores = pong_game.run(m_games=10, states=[])
        self.assertGreater(np.average(scores),12,
                           msg='''trained_model.pkl average score is %g which is not above 12
                           '''%(np.average(scores)))
        
    @weight(2)
    def test_deep_q_exceeds_16(self):
        import pong
        q_learner=submitted.deep_q(0.05,0.05,0.99,5)
        q_learner.load('trained_model.pkl')
        pong_game = pong.PongGame(learner=q_learner, visible=False)
        scores = pong_game.run(m_games=10, states=[])
        self.assertGreater(np.average(scores),16,
                           msg='''trained_model.pkl average score is %g which is not above 16
                           '''%(np.average(scores)))
        
    @weight(2)
    def test_deep_q_exceeds_18(self):
        import pong
        q_learner=submitted.deep_q(0.05,0.05,0.99,5)
        q_learner.load('trained_model.pkl')
        pong_game = pong.PongGame(learner=q_learner, visible=False)
        scores = pong_game.run(m_games=10, states=[])
        self.assertGreater(np.average(scores),18,
                           msg='''trained_model.pkl average score is %g which is not above 18
                           '''%(np.average(scores)))
        
    @weight(2)
    def test_deep_q_exceeds_20(self):
        import pong
        q_learner=submitted.deep_q(0.05,0.05,0.99,5)
        q_learner.load('trained_model.pkl')
        pong_game = pong.PongGame(learner=q_learner, visible=False)
        scores = pong_game.run(m_games=10, states=[])
        self.assertGreater(np.average(scores),20,
                           msg='''trained_model.pkl average score is %g which is not above 20
                           '''%(np.average(scores)))


