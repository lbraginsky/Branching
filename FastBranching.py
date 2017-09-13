'''
Created on Sep 3, 2017

@author: lbraginsky
'''

import numpy as np

class Branching(object):

    def __init__(self, branch_prob, fitness, initial_population, max_size):
        self.branch_prob = branch_prob
        self.fitness = fitness
        self.max_size = max_size
        self.members = initial_population
        self.generation = 0

    def step(self):
        # Everybody moves
        gen = self.members + np.random.normal(size=self.members.shape)
        # Everybody branches with luck
        ind = np.random.uniform(size=gen.shape[0]) < self.branch_prob(gen)
        gen = np.append(gen, gen[ind], axis=0)
        # Sort by fitness
        ind = np.argsort(self.fitness(gen))[::-1]
        # Truncate to max size
        # New generation becomes members
        self.members = gen[ind][:self.max_size]
        self.generation += 1

    def run(self, num_steps):
        for _i in range(num_steps):
            self.step()

    def stats(self):
        f = self.fitness(self.members)
        return {"count": len(f), "min": np.min(f), "max": np.max(f), "avg": np.mean(f), "std": np.std(f)}

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def simulation(ndim, steps_per_update):

    euclidean_fitness = lambda gen: np.sqrt(np.sum(gen**2, axis=1))
    sumsq_fitness = lambda gen: np.sum(gen**2, axis=1)
    sumabs_fitness = lambda gen: np.sum(abs(gen), axis=1)
    maxabs_fitness = lambda gen: np.max(abs(gen), axis=1)
    absprod_fitness = lambda gen: np.prod(abs(gen), axis=1)

    br = Branching(branch_prob=lambda x: 0.05,
                   fitness=euclidean_fitness,
                   initial_population=np.zeros(shape=(1, ndim)),
                   max_size=1000)

    if not ndim in [1, 2, 3]:
        raise NotImplementedError("Cannot display {}-dimensions".format(ndim))

    ax_lims = [(-50, 50)] * ndim
    def scaling():
        d_lims = list(zip(np.min(br.members, axis=0), np.max(br.members, axis=0)))
        if any(d[0] < a[0] or d[1] > a[1] for a, d in zip(ax_lims, d_lims)):
            r = max(b - a for a, b in d_lims) * 0.75
            for d in range(ndim):
                c = sum(d_lims[d])/2
                ax_lims[d] = (c - r, c + r)
        
    def plot():
        ax = fig.add_subplot(111, projection='3d' if ndim==3 else None)
        ax.set_autoscale_on(False)
        if ndim==1: ax.get_yaxis().set_visible(False)
        return ax

    def scatter_d1():
        x = [x[0] for x in br.members]
        y = [0 for v in x]
        ax.scatter(x, y, marker='.')
        ax.set_xlim(ax_lims[0])
        
    def scatter_d2():
        x, y = zip(*br.members)
        ax.scatter(x, y, marker='.')
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])

    def scatter_d3():
        x, y, z = zip(*br.members)
        ax.scatter(x, y, z, s=5)
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
        ax.set_zlim(ax_lims[2])

    scatter = [scatter_d1, scatter_d2, scatter_d3][ndim-1]

    fig = plt.figure(figsize=(9, 8))
    ax = plot()

    import time
    def update(i):
        t = time.time()
        br.run(steps_per_update)
        ax.clear()
        scaling()
        scatter()
        print("Generation: {}, stats: {}, time: {:.2}".
              format(br.generation, br.stats(), time.time() - t))

    ani = animation.FuncAnimation(fig, update, interval=1)
    plt.show()

simulation(ndim=2, steps_per_update=1)

import cProfile
import pstats

def profile(cmd):
    profileName = 'profile'
    cProfile.run(cmd, profileName)
    p = pstats.Stats(profileName)
    p.strip_dirs().sort_stats('time').print_stats()

# profile("simulation(ndim=2, steps_per_update=1)")
