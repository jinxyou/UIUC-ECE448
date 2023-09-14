# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021, 
# Inspired by previous work by Michael Abir (abir2@illinois.edu) and Rahul Kunji (rahulsk2@illinois.edu)

from collections import namedtuple
from itertools import chain 

class MazeError(Exception):
    pass

class Maze:
    """
    creates a maze instance given a `path` to a file containing characters in `legend`. 
    """
    def __init__(self, path, legend = {'wall': '%', 'start': 'P', 'waypoint': '.'}):

        # Passed in legend cannot introduce anything new
        for key in 'wall', 'start', 'waypoint':
            if key not in legend:
                raise ValueError('undefined legend key \'{0}\''.format(key))
        
        # Creates a legend to abstract away ASCII, e.g. self.legend.wall
        self.legend = namedtuple('legend', ('wall', 'start', 'waypoint'))(
            legend['wall'], 
            legend['start'], 
            legend['waypoint'])
        
        with open(path) as file:
            lines = tuple(line.strip() for line in file.readlines() if line)
        
        # Stores copy of ASCII maze in self._storage as well as dimensions in self.size.x/y
        n = len(lines)
        m = min(map(len, lines))
        
        if any(len(line) != m for line in lines):
            raise MazeError('(maze \'{0}\'): all maze rows must be the same length (shortest row has length {1})'.format(path, m))
        
        self._storage   = lines 
        self.size       = namedtuple('size', ('x', 'y'))(m, n)
        
        if any(self[x] != self.legend.wall for x in chain(
            ((    0, j) for j in range(m)), 
            ((n - 1, j) for j in range(m)), 
            ((i,     0) for i in range(n)), 
            ((i, m - 1) for i in range(n)))):
            raise MazeError('(maze \'{0}\'): maze borders must only contain `wall` cells (\'{1}\')'.format(path, self.legend.wall))
        if n < 3 or m < 3:
            raise MazeError('(maze \'{0}\'): maze dimensions ({1}, {2}) must be at least (3, 3)'.format(path, n, m))
        
        # Checks if only 1 start, if so, stores index in self.start
        self.start  = None 
        for x in ((i, j) 
            for i in range(self.size.y) 
            for j in range(self.size.x) if self[i, j] == self.legend.start):
            if self.start is None:
                self.start = x
            elif type(self.start) is int:
                self.start += 1 
            else: 
                self.start  = 2
        if type(self.start) is int or self.start is None:
            raise MazeError('(maze \'{0}\'): maze must contain exactly one `start` cell (\'{1}\') (found {2})'.format(
                path, self.legend.start, 0 if self.start is None else self.start))
        
        # Stores waypoint indices in self.waypoints
        self.waypoints = tuple((i, j) 
            for i in range(self.size.y) 
            for j in range(self.size.x) if self[i, j] == self.legend.waypoint)
        
        # there is no point in making this private since anyone trying to cheat 
        # could simply overwrite the underscored variable
        self.states_explored    = 0
    
    def __getitem__(self, index):
        """Access data at index via self[index] instead of using self._storage"""
        i, j = index
        if 0 <= i < self.size.y and 0 <= j < self.size.x:
            return self._storage[i][j]
        else:
            raise IndexError('cell index ({0}, {1}) out of range'.format(i, j))
    
    def indices(self):
        """Returns generator of all indices in maze"""
        return ((i, j) 
            for i in range(self.size.y) 
            for j in range(self.size.x))
    
    def navigable(self, i, j):
        """Check if moving to (i,j) is a valid move"""
        try:
            return self[i, j] != self.legend.wall 
        except IndexError:
            return False 

    def neighbors(self, i, j):
        """Returns list of neighboing squares that can be moved to from the given row,col"""
        self.states_explored += 1 
        return tuple(x for x in (
            (i + 1, j),
            (i - 1, j),
            (i, j + 1),
            (i, j - 1)) 
            if self.navigable( * x ))

    def validate_path(self, path):
        # validate type and shape 
        if len(path) == 0:
            return 'path must not be empty'
        if not all(len(vertex) == 2 for vertex in path):
            return 'each path element must be a two-element sequence'
        
        # normalize path in case student used an element type that is not `tuple` 
        path = tuple(map(tuple, path))

        # check if path is contiguous
        for i, (a, b) in enumerate(zip(path, path[1:])):
            if sum(abs(b - a) for a, b in zip(a, b)) != 1:
                return 'path vertex {1} ({4}, {5}) must be exactly one move away from path vertex {0} ({2}, {3})'.format(
                    i, i + 1, * a , * b )

        # check if path is navigable 
        for i, x in enumerate(path):
            if not self.navigable( * x ):
                return 'path vertex {0} ({1}, {2}) is not a navigable maze cell'.format(i, * x )
        
        # check if path ends at a waypoint 
        for waypoint in self.waypoints:
            if path[-1] == waypoint:
                break 
        else:
            return 'last path vertex {0} ({1}, {2}) must be a waypoint'.format(len(path) - 1, * path[-1] )

        # check for unnecessary path segments
        indices = {}
        for i, x in enumerate(path):
            if x in indices:
                if all(self[x] != self.legend.waypoint for x in path[indices[x] : i]):
                    return 'path segment [{0} : {1}] contains no waypoints'.format(indices[x], i)
            indices[x] = i 
        
        # check if path contains all waypoints 
        for i, x in enumerate(self.waypoints):
            if x not in indices:
                return 'waypoint {0} ({1}, {2}) was never visited'.format(i, * x )
