'''
Created on Sep 3, 2017

@author: lbraginsky
'''

import numpy as np

class Branching(object):

    def __init__(self, branch_prob, fitness, initial_population, max_size):
        self.branch_prob = branch_prob
        try:
            self.fitness, self.low_fit = fitness
        except TypeError:
            self.fitness, self.low_fit = fitness, False
        self.max_size = max_size
        self.members = initial_population
        self.generation = 0

    def step(self):
        # Everybody moves
        gen = self.members + np.random.normal(size=self.members.shape)
        # Everybody branches with luck
        ind = np.random.uniform(size=gen.shape[0]) < self.branch_prob
        gen = np.append(gen, gen[ind], axis=0)
        # Sort by fitness
        ind = np.argsort(self.fitness(gen))
        if not self.low_fit: ind = ind[::-1]
        # Truncate to max size
        # New generation becomes members
        self.members = gen[ind][:self.max_size]
        self.generation += 1

    def run(self, num_steps):
        for _i in range(num_steps):
            self.step()

    def stats(self):
        f = self.fitness(self.members)
        return {"gen": self.generation, "count": len(f), "min": np.min(f), "max": np.max(f), "avg": np.mean(f), "std": np.std(f)}

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def simulation(br, steps_per_update):

    ndim = br.members.shape[1]
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
        ax.scatter(x, y, marker='.', s=10)
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
        ax.annotate("Gen: {}".format(br.generation),
            xy=(1, 0), xycoords='axes fraction',
            xytext=(-10, 10), textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='bottom')

    def scatter_d3():
        x, y, z = zip(*br.members)
        ax.scatter(x, y, z, s=5)
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
        ax.set_zlim(ax_lims[2])

    scatter = [scatter_d1, scatter_d2, scatter_d3][ndim-1]

    fig = plt.figure(figsize=(9, 8))
    ax = plot()

    class Controls(object): pass
    controls = Controls()

    class Controller(object):
        def __init__(self):
            self.running = True
            self.steps = steps_per_update
        def toggle_running(self, event):
            if self.running:
                ani.event_source.stop()
            else:
                ani.event_source.start()
            self.running = not self.running
        def update_steps(self, event):
            controller.steps = int(controls.steps.val)

    controller = Controller()

    def update(i):
        import time        
        t = time.time()
        br.run(controller.steps)
        ax.clear()
        scaling()
        scatter()
        print("{}, time: {:.2}".format(br.stats(), time.time() - t))

    from matplotlib.widgets import Button, Slider
    controls.tgl = Button(plt.axes([0.8, 0.02, 0.1, 0.04]), "Stop/Go")
    controls.tgl.on_clicked(controller.toggle_running)
    controls.steps = Slider(plt.axes([0.55, 0.03, 0.2, 0.02]), 'Steps', 1, 100, valinit=controller.steps, valfmt='%.0f')
    controls.steps.on_changed(controller.update_steps)

    ani = animation.FuncAnimation(fig, update, interval=1, init_func=lambda: None)

    plt.show()

euclidean_fitness = lambda gen: np.sqrt(np.sum(gen**2, axis=1))
sumsq_fitness = lambda gen: np.sum(gen**2, axis=1)
sumabs_fitness = lambda gen: np.sum(abs(gen), axis=1)
maxabs_fitness = lambda gen: np.max(abs(gen), axis=1)
absprod_fitness = lambda gen: np.prod(abs(gen), axis=1)

def sparsity_fitness_fun(gen, radius):
    from scipy import spatial
    tree = spatial.cKDTree(gen)
    f = lambda p: len(tree.query_ball_point(p, r=radius))
    counts = np.apply_along_axis(f, 1, gen)
    return counts
sparsity_fitness_r50 = lambda gen: sparsity_fitness_fun(gen, 50), True

ndim = 2
simulation(Branching(branch_prob=0.05,
                     fitness=euclidean_fitness,
                     initial_population=np.zeros(shape=(1, ndim)),
                     max_size=1000),
           steps_per_update=1)
