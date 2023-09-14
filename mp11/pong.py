import random, argparse, copy, submitted, json
import numpy as np

class random_learner():
    def __init__(self, epsilon, paddle_speed):
        self.epsilon = epsilon
        self.paddle_speed = paddle_speed
    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice([-self.paddle_speed,0,self.paddle_speed])
        else:
            return None

class PongGame():
    def __init__(self, ball_speed=4, paddle_speed=8, learner=None,
                 visible=True, state_quantization=[10,10,2,2,10]):
        '''
        Create a new pong game, with a specified player.
        
        @params:
        ball_speed (scalar int) - average ball speed in pixels/frame
        paddle_speed (scalar int) - paddle moves 0, +paddle_speed, or -paddle_speed
        learner - can be None if the player is human.  If not None, should be an 
          object of type random_learner, submitted.q_learner, or submitted.deep_q.
        visible (bool) - should this game have an attached pygame window?
        state_quantization (list) - if not None, state variables are quantized
          into integers of these cardinalities before being passed to the learner.
        '''
        self.learner = learner
        self.game_h =  400
        self.game_w = 600
        self.radius = 20
        self.paddle_w = 8
        self.paddle_h = 80
        self.ball_speed = ball_speed
        self.max_ball_speed = 2*ball_speed
        self.paddle_speed = paddle_speed
        self.visible = visible
        self.half_pad_w = self.paddle_w / 2
        self.half_pad_h = self.paddle_h / 2
        self.score = 0
        self.n_games = 0
        self.max_score = 0
        self.state_quantization = state_quantization
        
        if learner==None and not(visible):
            raise RuntimeError('Human player can only be used with a visible playing board')

        # Initialize the visible display
        if self.visible:
            import pong_display
            self.display = pong_display.PongBoard(self.game_w,self.game_h,self.paddle_w,self.paddle_h)
            
    def ball_init(self):
        '''Spawn a ball in the center of the board with random velocity'''
        ball_x = self.game_w/2
        ball_y = self.game_h/2
        ball_vx = -self.ball_speed
        ball_vy = random.choice([-self.ball_speed,self.ball_speed])
        return ball_x, ball_y, ball_vx, ball_vy

    def state_init(self):
        '''Spawn a ball in the center of the board with random velocity'''
        ball_x, ball_y, ball_vx, ball_vy = self.ball_init()
        paddle_y = self.game_h/2
        return([ball_x, ball_y, ball_vx, ball_vy, paddle_y])

    def quantize_state(self, state):
        '''
        Quantize [ball_x, ball_y, ball_vx, ball_vy, paddle_y ] using 
        the number of levels, for each variable, specified in self.state_quantization.
        '''
        return [ int(state[0]*self.state_quantization[0]/self.game_w),
                 int(state[1]*self.state_quantization[1]/self.game_h),
                 int((state[2]+self.max_ball_speed)*self.state_quantization[2]/(2*self.max_ball_speed+1)),
                 int((state[3]+self.max_ball_speed)*self.state_quantization[3]/(2*self.max_ball_speed+1)),
                 int(state[4]*self.state_quantization[4]/self.game_h) ]


    def update(self, state, paddle_v):
        ball_x, ball_y, ball_vx, ball_vy, paddle_y = state
        reward = 0
        
        # ball collision with walls
        if ball_y <= self.radius:
            ball_vy = max(1,abs(ball_vy) + random.choice([-1,0,1]))
        if ball_y >= self.game_h - self.radius:
            ball_vy = min(-1,-abs(ball_vy) + random.choice([-1,0,1]))
        if ball_x <= self.radius:
            ball_vx = max(1,abs(ball_vx) + random.choice([-1,0,1]))
    
        #print('ball_x:',ball_x,', threshold: ',self.game_w-self.radius-self.paddle_w)
        if ball_x >= self.game_w - self.radius - self.paddle_w:
            # if ball hits squarely on paddle, reverse its velocity
            if abs(ball_y-paddle_y) <= self.half_pad_h:
                ball_vx = min(-1, - 1.1*abs(ball_vx) + random.choice([-1,0,1]))
                ball_vy = 1.1*ball_vy + random.choice([-1,0,1])
                self.score += 1
                reward = 1
                
            # if ball hits obliquely on paddle, reverse vx and vy
            elif abs(ball_y-paddle_y) <= self.half_pad_h + self.radius:
                ball_vx = min(-1, - 1.1*abs(ball_vy) + random.choice([-1,0,1]))
                ball_vy = np.sign(ball_y-paddle_y) * (1.1*abs(ball_vx)+random.choice([-1,0,1]))
                self.score += 1
                reward = 1
                
            # if ball is past the paddle, reset game
            else:
                self.score = 0
                self.n_games += 1
                ball_x, ball_y, ball_vx, ball_vy = self.ball_init()
                reward = -10
            #print('ball at end of field, reward=%d, score=%d'%(reward,self.score))
                
        # paddle position
        paddle_y = max(self.half_pad_h, min(self.game_h-self.half_pad_h, paddle_y + paddle_v))
        
        # ball
        if abs(ball_vx) > self.max_ball_speed:
            ball_vx = self.max_ball_speed * np.sign(ball_vx)
        if abs(ball_vy) > self.max_ball_speed:
            ball_vy = self.max_ball_speed * np.sign(ball_vy)
        ball_x = max(self.radius, ball_x+ball_vx)
        ball_y = max(self.radius, min(self.game_h-self.radius, ball_y+ball_vy))
        return [ball_x, ball_y, ball_vx, ball_vy, paddle_y], reward

    def run(self, m_rewards=np.inf, m_games=np.inf, m_frames=np.inf, states=[]):
        '''
        Run the game.
        @param
        m_frames (scalar int): maximum number of frames to be played
        m_rewards (scalar int): maximum number of rewards earned (+ or -)
        m_games (scalar int): maximum number of games
        states (list): list of states whose Q-values should be returned
           each state is a list of 5 ints: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
           and the y-position of the paddle, all quantized.
           0 <= state[i] < state_cardinality[i], for all i in [0,4].

          
        @return
        scores (list): list of scores of all completed games
        
        The following will be returned only if the player is q_learning or deep_q.
        New elements will be added to these lists once/frame if m_frames is specified,
        else once/reward if m_rewards is specified, else once/game:
          q_achieved (list): list of the q-values of the moves that were taken
          q_states (list): list of the q-values of requested states
        '''
        state = self.state_init()
        paddle_v = 0

        # Main loop
        # Until the game is killed, draw the game board, then update game state
        n_rewards = 0
        n_games = 0
        n_frames = 0
        scores = [ 0 ]
        q_achieved = []
        q_states = []
        
        while n_rewards < m_rewards and n_games < m_games and n_frames < m_frames:
            n_frames += 1
            
            # Draw the game board
            if self.visible:
                self.display.draw(state[0],state[1],state[4],self.radius)
                self.display.draw_scores(self.n_games,self.score,self.max_score)
                self.display.update_display()

            # Get an action from the player: human, Q-learner, or random
            if self.learner==None and self.visible:
                action = self.display.get_event()
            elif self.learner==None:
                raise RuntimeError('pong learner has unrecognized type:',type(self.learner))
            elif self.state_quantization==None:
                action = self.learner.act(state)
            else:
                action = self.learner.act(self.quantize_state(state))    
            if action != None:
                paddle_v = action * self.paddle_speed

            # One move of game play
            scores[-1] = self.score
            self.max_score = max(self.score, self.max_score)
            newstate, reward = self.update(state, paddle_v)
            if reward != 0:
                n_rewards += 1
                if reward < 0: 
                    print('Completed %d games, %d rewards, %d frames, score %d, max score %d'%(n_games,n_rewards,n_frames, scores[-1], self.max_score))
                    n_games += 1
                    if n_games < m_games:
                        scores.append([0])
                    
            # learn
            if type(self.learner)==submitted.q_learner or type(self.learner)==submitted.deep_q:
                if self.state_quantization:
                    state = self.quantize_state(state)
                    self.learner.learn(state, action, reward, self.quantize_state(newstate))
                else:
                    self.learner.learn(state, action, reward, newstate)
                if m_frames<np.inf or (m_rewards<np.inf and reward!=0) or (reward<0):
                    q_achieved.append(self.learner.report_q(state))
                    q_states.append(np.array([self.learner.report_q(s) for s in states]))
            state = newstate
            
        # Return value depends on learner type
        if type(self.learner)==submitted.q_learner or type(self.learner)==submitted.deep_q:
            return scores, q_achieved, q_states
        else:
            return scores


################################################################################################
# Command line arguments
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description     = 'pong.py - one-player pong',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--player', default = 'human',
                        choices = ('random', 'human', 'q_learning','deep_q'),
                        help = 'Is player a human, a random player, or some type of AI?')
    parser.add_argument('--ball_speed', default = '4',
                        help = 'How fast should the ball be (on average)?')
    parser.add_argument('--paddle_speed', default = '8',
                        help = 'How fast should the paddle be?')
    parser.add_argument('--alpha', default = '0.05',
                        help = 'Reinforcement learning rate')
    parser.add_argument('--epsilon', default = '0.1',
                        help = 'Exploration probability')
    parser.add_argument('--gamma', default = '0.99',
                        help = 'discount factor')
    parser.add_argument('--nfirst', default = '5',
                        help = 'number of exploration per state/action')
    parser.add_argument('--state_quantization', default='[10,10,2,2,10]',
                        help='''
                        Number of integer levels to which each of the five state variables is 
                        quantized when passed to q_learning.  No quantization is applied when
                        the state is passed to deep_q.
                        ''')
    args   = parser.parse_args()
    alpha = float(args.alpha)
    epsilon = float(args.epsilon)
    gamma = float(args.gamma)
    nfirst = int(args.nfirst)
    ball_speed = float(args.ball_speed)
    paddle_speed = float(args.paddle_speed)
    state_quantization = [ int(x) for x in json.loads(args.state_quantization) ]
    
    if args.player=='q_learning':
        learner = submitted.q_learner(alpha,epsilon,gamma,nfirst,state_quantization)
    elif args.player=='deep_q':
        learner = submitted.deep_q(alpha,epsilon,gamma,nfirst)
        state_quantization = None
    elif args.player=='random':
        learner=random_learner(epsilon, paddle_speed)
        state_quantization = None
    else:
        learner=None
        state_quantization = None

    # Create and run the application
    application = PongGame(ball_speed=ball_speed,
                           paddle_speed=paddle_speed,
                           learner=learner,
                           state_quantization=state_quantization)
    application.run()
