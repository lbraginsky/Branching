'''
Created on Sep 3, 2017

@author: lbraginsky
'''

from random import random, gauss

def brownian_step(x):
    return tuple(v + gauss(0, 1) for v in x)

class Branching(object):

    def __init__(self, branch_prob, fitness, initial_population, max_size):
        self.branch_prob = branch_prob
        self.fitness = fitness
        self.max_size = max_size
        self.members = list(initial_population)
        self.generation = 0

    def step(self):
        # Everybody moves
        gen = [brownian_step(x) for x in self.members]
        # Everybody branches with luck
        gen.extend([x for x in gen if random() < self.branch_prob])
        # Sort by fitness
        gen.sort(key=self.fitness, reverse=True)
        # Truncate to max size
        del gen[self.max_size:]
        # New generation becomes members
        self.members = gen
        self.generation += 1

    def run(self, num_steps):
        for _i in range(num_steps):
            self.step()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

def simulation(ndim, steps_per_update):

    euclidean_fitness = lambda x: sqrt(sum(v*v for v in x))

    bm = Branching(branch_prob=0.05,
                   fitness=euclidean_fitness,
                   initial_population=[(0,)*ndim],
                   max_size=1000)

    if not ndim in [1, 2, 3]:
        raise NotImplementedError("Cannot display {}-dimensions".format(ndim))

    def plot():
        ax = fig.add_subplot(111, projection='3d' if ndim==3 else None)
        if ndim==1: ax.get_yaxis().set_visible(False)
        return ax

    def scatter_d1():
        x = [x[0] for x in bm.members]
        y = [0 for v in x]
        ax.scatter(x, y, marker='.')
        
    def scatter_d2():
        x, y = zip(*bm.members)
        ax.scatter(x, y, marker='.')

    def scatter_d3():
        x, y, z = zip(*bm.members)
        ax.scatter(x, y, z, s=5)

    scatter = [scatter_d1, scatter_d2, scatter_d3][ndim-1]

    fig = plt.figure(figsize=(9, 8))
    ax = plot()

    def update(i):
        bm.run(steps_per_update)
        ax.clear()
        scatter()
        print("Generation: {}, members: {}".format(bm.generation, len(bm.members)))

    ani = animation.FuncAnimation(fig, update, interval=1)
    plt.show()

simulation(ndim=3, steps_per_update=10)
