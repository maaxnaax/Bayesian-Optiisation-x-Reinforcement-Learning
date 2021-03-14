# import argparse
import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel

# FOR BAYES:
import neat
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
# from bayesian_optimization_util import plot_approximation, plot_acquisition
from bayes_plots import *
from bayes_acquisitions import *
from gym_solver import simulate_species, worker_evaluate_genome
# The name of the game to solve
game_name = 'CartPole-v0'


class args:

    # max - steps = 1000 - -episodes = 10 - -generations = 50 - -render
    render = False
    checkpoint = False

    max_steps = 5 # 1000
    episodes = 10
    generations = 5 # 50
    numCores = 4


def evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    env = gym.make(game_name)
    return simulate_species(net, env, args.episodes, args.max_steps, render=args.render)

def eval_fitness(genomes):
    for g in genomes:
        fitness = evaluate_genome(g)
        g.fitness = fitness


def f_train_network(a=0.0): # returns number of generations until solved



    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'gym_config')

    ret_list = list()
    # for i in range(5):
    for i in range(2):
        pop = population.Population(config_path)
        pop.config.prob_add_conn = a

        # Load checkpoint
        if args.checkpoint:
            pop.load_checkpoint(args.checkpoint)
        # Start simulation
        if args.render:
            pop.run(eval_fitness, args.generations)
        else:
            pe = parallel.ParallelEvaluator(args.numCores, worker_evaluate_genome)
            pop.run(pe.evaluate, args.generations)


        ret_list.append(-len(pop.statistics.generation_statistics))

        # print(str(ret_list))
    # print("THIS IS THE THING BEING ADDED TO THE LIST::  " + str(np.mean(ret_list) / args.generations)) )
    # print(np.mean(ret_list) / args.generations)

    return np.mean(ret_list)/args.generations

def run():
    noise = 0.05

    x1 = 0.33
    x2 = 0.66

    y1 = f_train_network(x1)
    y2 = f_train_network(x2)

    X_init = np.array([[x1], [x2]])
    Y_init = np.array([[y1], [y2]])

    bounds = np.array([[0, 1.0]])

    # # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

    # Gaussian process with Matérn kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)  # rbf generalisation
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)

    # Initialize samples
    X_sample = X_init
    Y_sample = Y_init

    # Number of iterations
    n_iter = 10

    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)

        # Obtain next noisy sample from the objective function
        Y_next = f_train_network(X_next)

        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(gpr, X, X_sample, Y_sample, X_next, show_legend=i == 0)
        plt.title(f'Iteration {i + 1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i == 0)


        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
    plt.show()




my_env = gym.make(game_name)
print("Input Nodes: %s" % str(len(my_env.observation_space.high)))
print("Output Nodes: %s" % str(my_env.action_space.n))



if __name__ == '__main__':

    run()

