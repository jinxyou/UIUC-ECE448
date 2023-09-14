import pygame, sys
import numpy as np
from pygame.locals import *


class PongBoard():
    def __init__(self, game_w, game_h, paddle_w, paddle_h):
        pygame.init()
        self.fps = pygame.time.Clock()
        self.window = pygame.display.set_mode([int(game_w),int(game_h)], 0, 32)
        pygame.display.set_caption('Pong')
        self.font = pygame.font.SysFont("Comic Sans MS", 20)
        self.paddle_corners = np.array([[-paddle_w/2, -paddle_h/2],
                                        [paddle_w/2, -paddle_h/2],
                                        [paddle_w/2, paddle_h/2],
                                        [-paddle_w/2, paddle_h/2],
                                        [-paddle_w/2, -paddle_h/2]])
        self.corners = np.array([[int(game_w),0],[0,0],[0,int(game_h)],
                                 [int(game_w),int(game_h)]])
        self.centerline = np.array([[int(game_w/2),0],[int(game_w/2),int(game_h)]])
        self.centerpoint = np.array([int(game_w/2),int(game_h/2)])
        self.paddle_x = game_w + 1 - paddle_w/2
        
        
    def draw_scores(self, n_games, score, max_score):
        '''Write current scores on the game board'''
        labels = ['Game ', 'Score ', 'Max Score ']
        scores = [n_games, score, max_score]
        pos = [10,30,50]
        for s,l,p in zip(scores,labels,pos):
            lab = self.font.render(l+str(s), 1, pygame.Color('yellow'))
            self.window.blit(lab, (470, p))
            
    def draw(self, ball_x, ball_y, paddle_y, ball_radius):
        '''Draw the game board'''
        self.window.fill(pygame.Color('black'))
        pygame.draw.line(self.window, pygame.Color('white'),self.centerline[0], self.centerline[1], 1)
        pygame.draw.line(self.window, pygame.Color('white'), self.corners[0], self.corners[1], 1)
        pygame.draw.line(self.window, pygame.Color('white'), self.corners[1], self.corners[2], 1)
        pygame.draw.line(self.window, pygame.Color('white'), self.corners[2], self.corners[3], 1)
        pygame.draw.circle(self.window, pygame.Color('white'), self.centerpoint, 70, 1)

        # Draw the paddle, ball, and score
        paddle_corners = [[int(self.paddle_x+c[0]), int(paddle_y+c[1])]
                          for c in self.paddle_corners]
        pygame.draw.polygon(self.window, pygame.Color('green'), paddle_corners, 0)
        pygame.draw.circle(self.window, pygame.Color('red'),
                           [int(ball_x),int(ball_y)],int(ball_radius),0)
    
    def get_event(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    return -1
                elif event.key == K_DOWN:
                    return 1
            elif event.type == KEYUP:
                return 0
            elif event.type == QUIT:
                pygame.quit()
                sys.exit()

    def update_display(self):
        pygame.display.update()
        self.fps.tick(60)


