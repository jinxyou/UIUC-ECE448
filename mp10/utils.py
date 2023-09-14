# utils.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dawei Sun (daweis2@illinois.edu) on 03/18/2023
"""
This file provides functions for loading and plotting the MDP examples
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
import json

class GridWorld:
    """
    A class to represent a grid world.
    """
    def __init__(self, filename):
        """
        Constructs an object from a file.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        self.W = np.array(data['wall']).astype(bool)
        self.T = np.array(data['isterminal']).astype(bool)
        self.R = np.array(data['rewards'])
        self.D = np.array(data['disturbances'])
        self.gamma = np.array(data['gamma'])
        self.M, self.N = self.R.shape

    def visualize(self, U=None):
        """
        This function visualizes the shape, the wall, and the terminal states of the environment. If a utility function U is provided, then it visualizes the utility function instead.
        """
        fig, ax = plt.subplots()
        ax.set_axis_off()
        tb = Table(ax, bbox=[0,0,1,1])

        nrows, ncols = self.R.shape
        width, height = 1.0 / ncols, 1.0 / nrows

        for (i, j), isterminal in np.ndenumerate(self.T):
            text = ' '
            if isterminal:
                text = '+1' if self.R[i, j] > 0 else '-1'
            if U is not None:
                text = '%.3f'%U[i, j]
            if self.W[i, j]:
                text = 'x'
            tb.add_cell(i, j, width, height, text=text, loc='center')

        ax.add_table(tb)
        plt.show()

def load_MDP(filename):
    return GridWorld(filename)
