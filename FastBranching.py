'''
Created on Sep 3, 2017

@author: lbraginsky
'''

import numpy as np

class Branching(object):

    @staticmethod
    def sparsity_fitness(gen, diameter):
        from scipy import spatial
        tree = spatial.cKDTree(gen)
        f = lambda p: len(tree.query_ball_point(p, r=0.5*diameter))
        return np.apply_along_axis(f, 1, gen)

    fitness_functions = {
        "euclidean":    lambda gen: np.sqrt(np.sum(gen**2, axis=1)),
        "sumsq":        lambda gen: np.sum(gen**2, axis=1),
        "sumabs":       lambda gen: np.sum(abs(gen), axis=1),
        "maxabs":       lambda gen: np.max(abs(gen), axis=1),
        "absprod":      lambda gen: np.prod(abs(gen), axis=1),
        "sparsity(100)": (lambda gen: Branching.sparsity_fitness(gen, 100), True)
        }

    def __init__(self, branch_prob, fitness, initial_population, max_size):
        self.branch_prob = branch_prob
        self.set_fitness(fitness)
        self.max_size = max_size
        self.members = initial_population
        self.generation = 0

    def set_fitness(self, fitness):
        f = Branching.fitness_functions[fitness]
        try:
            self.fitness_fun, self.low_fit = f
        except TypeError:
            self.fitness_fun, self.low_fit = f, False
        self.fitness = fitness
        
    def step(self):
        # Everybody moves
        gen = self.members + np.random.normal(size=self.members.shape)
        # Everybody branches with luck
        ind = np.random.uniform(size=gen.shape[0]) < self.branch_prob
        gen = np.append(gen, gen[ind], axis=0)
        # Sort by fitness
        ind = np.argsort(self.fitness_fun(gen))
        if not self.low_fit: ind = ind[::-1]
        # Truncate to max size
        # New generation becomes members
        self.members = gen[ind][:self.max_size]
        self.generation += 1

    def run(self, num_steps):
        for _i in range(num_steps):
            self.step()

    def stats(self):
        f = self.fitness_fun(self.members)
        return {"gen": self.generation, "count": len(f), "min": np.min(f), "max": np.max(f), "avg": np.mean(f), "std": np.std(f)}

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class Simulation(object):

    def __init__(self, branching, steps_per_update=1):
        ndim = branching.members.shape[1]
        if not ndim in [1, 2, 3]:
            raise NotImplementedError("Cannot display {}-dimensions".format(ndim))
        self.branching = branching
        self.steps_per_update = steps_per_update
        self.ndim = ndim
        self.running = False
        self.ax_lims = [(-50, 50)] * ndim
        self.scatter = [self.scatter_d1, self.scatter_d2, self.scatter_d3][ndim-1]
        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.plot()
        self.setup_controls()
    
    def scaling(self):
        members = self.branching.members
        d_lims = list(zip(np.min(members, axis=0), np.max(members, axis=0)))
        if any(d[0] < a[0] or d[1] > a[1] for a, d in zip(self.ax_lims, d_lims)):
            r = max(b - a for a, b in d_lims) * 0.75
            for d in range(len(d_lims)):
                c = sum(d_lims[d])/2
                self.ax_lims[d] = (c - r, c + r)
        
    def plot(self):
        ax = self.fig.add_subplot(111, projection='3d' if ndim==3 else None)
        ax.set_autoscale_on(False)
        if ndim==1: ax.get_yaxis().set_visible(False)
        return ax

    def scatter_d1(self):
        x = [x[0] for x in self.branching.members]
        y = [0 for _v in x]
        self.ax.scatter(x, y, marker='.')
        self.ax.set_xlim(self.ax_lims[0])
        
    def scatter_d2(self):
        x, y = zip(*self.branching.members)
        self.ax.scatter(x, y, marker='.', s=10)
        self.ax.set_xlim(self.ax_lims[0])
        self.ax.set_ylim(self.ax_lims[1])
        self.ax.annotate("Gen: {}".format(self.branching.generation),
            xy=(1, 0), xycoords='axes fraction',
            xytext=(-10, 10), textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='bottom')

    def scatter_d3(self):
        x, y, z = zip(*self.branching.members)
        self.ax.scatter(x, y, z, s=5)
        self.ax.set_xlim(self.ax_lims[0])
        self.ax.set_ylim(self.ax_lims[1])
        self.ax.set_zlim(self.ax_lims[2])

    def setup_controls(self):
        from matplotlib.widgets import Button, RadioButtons, Slider
        self.g_branch_prob = Slider(plt.axes([0.1, 0.03, 0.1, 0.02]), 'Branch\nprob', 0, 1,
                              valinit=self.branching.branch_prob, valfmt='%.2f')
        self.g_branch_prob.on_changed(self.update_branch_prob)
        self.g_max_size = Slider(plt.axes([0.3, 0.03, 0.1, 0.02]), 'Max\nsize', 1, 5,
                              valinit=np.log10(self.branching.max_size), valfmt='%.0f')
        self.g_max_size.valtext.set_text(self.branching.max_size)
        self.g_max_size.on_changed(self.update_max_size)
        self.g_fitness = RadioButtons(plt.axes([0.5, 0.01, 0.15, 0.13]), Branching.fitness_functions, self.branching.fitness)
        self.g_fitness.on_clicked(self.update_fitness)
        self.g_steps = Slider(plt.axes([0.8, 0.03, 0.1, 0.02]), 'Steps per\nupdate', 0, 3,
                              valinit=np.log10(self.steps_per_update), valfmt='%.0f')
        self.g_steps.valtext.set_text(self.steps_per_update)
        self.g_steps.on_changed(self.update_steps)
        self.g_tgl = Button(plt.axes([0.92, 0.1, 0.07, 0.04]), "Stop/Go")
        self.g_tgl.on_clicked(self.toggle_running)
        
    def update_branch_prob(self, event):
        self.branching.branch_prob = self.g_branch_prob.val

    def update_max_size(self, event):
        self.branching.max_size = int(10**self.g_max_size.val)
        self.g_max_size.valtext.set_text(self.branching.max_size)

    def update_fitness(self, label):
        self.branching.set_fitness(label)

    def update_steps(self, event):
        self.steps_per_update = int(10**self.g_steps.val)
        self.g_steps.valtext.set_text(self.steps_per_update)

    def toggle_running(self, event):
        if self.running:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()
        self.running = not self.running

    def update(self, i):
        import time        
        t = time.time()
        self.branching.run(self.steps_per_update)
        self.ax.clear()
        self.scaling()
        self.scatter()
        print("{}, time: {:.2}".format(self.branching.stats(), time.time() - t))

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, init_func=lambda: None)
        self.running = True
        plt.show()

ndim = 2
simulation = Simulation(Branching(branch_prob=0.05,
                                  fitness="euclidean",
                                  initial_population=np.zeros(shape=(1, ndim)),
                                  max_size=1000),
                        steps_per_update=1)
simulation.run()
