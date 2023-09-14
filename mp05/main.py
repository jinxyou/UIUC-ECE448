# main.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021, 
# Inspired by previous work by Michael Abir (abir2@illinois.edu) and Rahul Kunji (rahulsk2@illinois.edu)

"""
This file contains the main application that is run for this MP. It
initializes the pygame context, and handles the interface between the
game and the search algorithm.
"""

import sys, argparse, time

import pygame

from maze import Maze
import submitted

class gradient:
    def __init__(self, start, end):
        # rgb colors
        self.start  = start 
        self.end    = end 
    
    def __getitem__(self, fraction):
        t = fraction[0] / max(1, fraction[1] - 1) # prevent division by zero
        return tuple(max(0, min(start * (1 - t) + end * t, 255)) 
            for start, end in zip(self.start, self.end))

class agent:
    def __init__(self, position, maze):
        self.position   = position 
        self.maze       = maze 

    def move(self, move):
        position = tuple(i + move for i, move in zip(self.position, move))
        if self.maze.navigable( * position ):
            previous        = self.position
            self.position   = position 
            return previous,
        else: 
            return ()
            
class Application:
    def __init__(self, human = True, scale = 20, fps = 30, alt_color = False):
        self.running    = True
        self.scale      = scale
        self.fps        = fps
        
        self.human      = human 
        # accessibility for colorblind students 
        if alt_color:
            self.gradient = gradient((64, 224, 208), (139, 0, 139))
        else:
            self.gradient = gradient((255, 0, 0), (0, 255, 0))

    def run(self, filepath, mode, save):
        self.maze   = Maze(filepath)
        
        self.window = tuple(x * self.scale for x in self.maze.size)

        if self.human:
            self.agent = agent(self.maze.start, self.maze)
            
            path            = []
            states_explored = 0
        else:
            #time in seconds
            time_start      = time.time()
            
            path            = getattr(submitted, mode)(self.maze)
            states_explored = self.maze.states_explored
            
            time_total      = time.time() - time_start   

        pygame.init()
        
        self.surface = pygame.display.set_mode(self.window, pygame.HWSURFACE)
        self.surface.fill((255, 255, 255))
        pygame.display.flip()
        pygame.display.set_caption('MP1 ({0})'.format(filepath))

        if self.human:
            self.draw_player()
        else:
            print("""
Results 
{{
    path length         : {0}
    states explored     : {1}
    total execution time: {2:.2f} seconds
}}
            """.format(len(path), states_explored, time_total))
            
            self.draw_path(path)

        self.draw_maze()
        self.draw_start()
        self.draw_waypoints()

        pygame.display.flip()
        
        if type(save) is str:
            pygame.image.save(self.surface, save)
            self.running = False
        
        clock = pygame.time.Clock()
        
        while self.running:
            pygame.event.pump()
            clock.tick(self.fps)
            
            for event in pygame.event.get():
                if      event.type == pygame.QUIT:
                    raise SystemExit
                elif    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit
                elif    event.type == pygame.KEYDOWN and self.human:
                    try:
                        move = {
                            pygame.K_RIGHT  : ( 0,  1),
                            pygame.K_LEFT   : ( 0, -1),
                            pygame.K_UP     : (-1,  0),
                            pygame.K_DOWN   : ( 1,  0),
                        }[event.key] 
                        path.extend(self.agent.move(move))
                    except KeyError: 
                        pass
                
                    self.loop(path + [self.agent.position])

    # The game loop is where everything is drawn to the context. Only called when a human is playing
    def loop(self, path):
        self.draw_path(path)
        self.draw_waypoints()
        self.draw_player()
        pygame.display.flip()

    # Draws the path (given as a list of (row, col) tuples) to the display context
    def draw_path(self, path):
        for i, x in enumerate(path):
            self.draw_square( * x , self.gradient[i, len(path)])
    
    # Draws the full maze to the display context
    def draw_maze(self):
        for x in self.maze.indices():
            if self.maze[x] == self.maze.legend.wall: 
                self.draw_square( * x )
    
    def draw_square(self, i, j, color = (0, 0, 0)):
        pygame.draw.rect(self.surface, color, tuple(i * self.scale for i in (j, i, 1, 1)), 0)
    
    def draw_circle(self, i, j, color = (0, 0, 0), radius = None):
        if radius is None:
            radius = self.scale / 4
        pygame.draw.circle(self.surface, color, tuple(int((i + 0.5) * self.scale) for i in (j, i)), int(radius))

    # Draws the player to the display context, and draws the path moved (only called if there is a human player)
    def draw_player(self):
        self.draw_circle( * self.agent.position , (0, 0, 255))

    # Draws the waypoints to the display context
    def draw_waypoints(self):
        for x in self.maze.waypoints:
            self.draw_circle( * x )

    # Draws start location of path
    def draw_start(self):
        i, j = self.maze.start
        pygame.draw.rect(self.surface, (0, 0, 255), tuple(int(i * self.scale) for i in (j + 0.25, i + 0.25, 0.5, 0.5)), 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description     = 'CS440 MP1 Search', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path',
                        help = 'path to maze file')
    parser.add_argument('--search', dest = 'search', type = str, default = 'bfs',
                        choices = ('bfs', 'astar_corner', 'astar_single', 'fast', 'astar_multiple'), 
                        help = 'search method')
    parser.add_argument('--scale',  dest = 'scale', type = int, default = 20,
                        help = 'display scale')
    parser.add_argument('--fps',    dest = 'fps', type = int, default = 30,
                        help = 'display framerate')
    parser.add_argument('--human', default = False, action = 'store_true',
                        help = 'run in human-playable mode')
    parser.add_argument('--save', dest = 'save', type = str, default = None,
                        help = 'save output to image file')
    parser.add_argument('--altcolor', dest = 'altcolor', default = False, action = 'store_true',
                        help = 'view in an alternate color scheme')

    arguments   = parser.parse_args()
    application = Application(arguments.human, arguments.scale, arguments.fps, arguments.altcolor)
    application.run(
        filepath    = arguments.path, 
        mode        = arguments.search, 
        save        = arguments.save)
