# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Implement bfs function
    root = maze.start
    # print("root:", root)
    goal = maze.waypoints[0]
    # print("goal:", goal)
    Q = queue.Queue()
    explored = ()
    explored += (root,)
    Q.put(root)
    paths = [[root]]
    while Q.empty() is False:
        v = Q.get()
        # print("v:",v)
        if v == goal:
            for path in paths:
                if path[-1] == v:
                    return path
        for w in maze.neighbors(v[0], v[1]):
            # print("w:",w)
            if w not in explored:
                explored += (w,)
                for path in paths:
                    if path[-1] == v:
                        paths.append(path + [w])
                Q.put(w)
        # print("paths:", paths)


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Implement astar_single

    start = maze.start
    goal = maze.waypoints[0]
    Q = {start}
    cameFrom = dict()
    g = dict()
    for indice in list(maze.indices()):
        g[indice] = float('inf')
    g[start] = 0

    def h(a):
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    f = dict()
    for indice in list(maze.indices()):
        f[indice] = float('inf')
    f[start] = h(start)

    def reconstruct_path(cameFrom, curr):
        path = [curr]
        # print("path:", path)
        while curr in cameFrom.keys():
            curr = cameFrom[curr]
            # print("curr:", curr)
            path.insert(0, curr)
        return path

    while Q:
        # print("cameFrom:", cameFrom)
        curr = min(Q, key=f.get)
        if curr == goal:
            return reconstruct_path(cameFrom, curr)

        Q.remove(curr)
        for neighbor in maze.neighbors(curr[0], curr[1]):
            tentative_g = g[curr] + 1
            if tentative_g < g[neighbor]:
                cameFrom[neighbor] = curr
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + h(neighbor)
                if neighbor not in Q:
                    Q.add(neighbor)


# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """




    return []
