# import argparse
import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel
from gym_solver import simulate_species, worker_evaluate_genome

# FOR BAYES:
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from GPyOpt.methods import ModularBayesianOptimization

### User Params ###

# The name of the game to solve
game_name = 'CartPole-v0'

import time
start_time = time.time()

class args:

    # max - steps = 1000 - -episodes = 10 - -generations = 50 - -render
    render = False
    checkpoint = False

    max_steps = 10000
    episodes = 100
    generations = 150 # 50  #  # 50
    numCores = 8


### End User Params ###


def evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    env = gym.make(game_name)
    return simulate_species(net, env, args.episodes, args.max_steps, render=args.render)

def eval_fitness(genomes):
    for g in genomes:
        fitness = evaluate_genome(g)
        g.fitness = fitness

# GAME

list_of_scores = list()
list_of_variances = list()
# list_ten_variances = list()
list_of_ret_list = list()

def f_train_network_mean_num_gens(parameters): # returns number of generations until solved

    # print('This is parameters:  '+str(parameters))
    a = parameters[0][0]
    b = parameters[0][1]
    c = parameters[0][2]
    d = parameters[0][3]

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'gym_config')

    ret_list = list()
    # ten_variances = list()

    for i in range(10):

        pop = population.Population(config_path)

        # Connection version of the expereriment:
        pop.config.prob_add_conn = a
        pop.config.prob_delete_conn = b

        # # Node version of experiment:
        pop.config.prob_add_node = c
        pop.config.prob_delete_node = d

        # Load checkpoint
        if args.checkpoint:
            pop.load_checkpoint(args.checkpoint)
        # Start simulation
        if args.render:
            pop.run(eval_fitness, args.generations)
        else:
            pe = parallel.ParallelEvaluator(args.numCores, worker_evaluate_genome)
            pop.run(pe.evaluate, args.generations)


        ret_list.append(len(pop.statistics.generation_statistics))


    # list_of_variances.append(np.std(ret_list))
    list_of_ret_list.append(ret_list)
    score = np.mean(ret_list)/args.generations
    # list_of_scores.append(score)
    return np.array(score)

# # Hyperparameters to tune and their ranges
# param_dist = {"learning_rate": uniform(0, 1),
#               "gamma": uniform(0, 5),
#               "max_depth": range(1,50),
#               "n_estimators": range(1,300),
#               "min_child_weight": range(1,10)}
#
# rs = RandomizedSearchCV(xgb, param_distributions=param_dist,
#                         scoring='neg_mean_squared_error', n_iter=25)
#
# # Run random search for 25 iterations
# rs.fit(X, Y);




# # Optimization objective
# def cv_score(parameters):
#     parameters = parameters[0]
#     score = cross_val_score(
#                 XGBRegressor(learning_rate=parameters[0],
#                               gamma=int(parameters[1]),
#                               max_depth=int(parameters[2]),
#                               n_estimators=int(parameters[3]),
#                               min_child_weight = parameters[4]),
#                 X, Y, scoring='neg_mean_squared_error').mean()
#     score = np.array(score)
#     return score

my_env = gym.make(game_name)



#
# print("Input Nodes: %s" % str(len(my_env.observation_space.high)))
# print("Output Nodes: %s" % str(my_env.action_space.n))

if __name__ == '__main__':

    bds = [{'name': 'prob_add_connection', 'type': 'continuous', 'domain': (0, 1)},
           {'name': 'prob_delete_connection', 'type': 'continuous', 'domain': (0, 1)},
           {'name': 'prob_add_node', 'type': 'continuous', 'domain': (0, 1)},
           {'name': 'prob_delete_node', 'type': 'continuous', 'domain': (0, 1)}
           ]

    # optimizer = BayesianOptimization(f=f_train_network_mean_num_gens,
    #                                  domain=bds,
    #                                  model_type='GP',
    #                                  acquisition_type='EI',
    #                                  acquisition_jitter=0.05,
    #                                  exact_feval=True,
    #                                  maximize=True)
    # # Only 20 iterations because we have 5 initial random points
    # optimizer.run_optimization(max_iter=50, save_models_parameters=True, verbosity=True)
    # print("=" * 20)
    # print("Value of (x,y) that minimises the objective:" + str(optimizer.x_opt))
    # print("Minimum value of the objective: " + str(optimizer.fx_opt))
    # print("=" * 20)
    #
    # optimizer.plot_acquisition()
    # # Plot some more characteristics:
    #
    # optimizer.plot_convergence()  # Can clearly see it spends quite some time exploring the best small section
    # # which it thinks is the best space

    # ==============================================================
    # Trying MODULAR BO
    # ==============================================================

    # Determine the subset where we are allowed to sample
    feasible_region = GPyOpt.Design_space(space=bds)
    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)

    # CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(f_train_network_mean_num_gens)

    # CHOOSE the model type
    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)

    # CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    # CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

    # CHOOSE a collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # Now create BO object
    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator,
                                                    initial_design)

    # --- Stop conditions
    max_time = None
    max_iter = 100
    tolerance = 1e-8  # distance between two consecutive observations
    # if we're sampling a region in such fine detail then it is likely that we've found the true min.

    # Run the optimization
    bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False)

    # SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE
    # SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE
    # SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE
    import json
    a = list_of_ret_list
    # a = [1, 2, 3]
    with open('list_of_ret_list.txt', 'w') as f:
        f.write(json.dumps(a))

    BO_time = [time.time() - start_time]

    # a = [1, 2, 3]
    with open('BO_time.txt', 'w') as f:
        f.write(json.dumps(BO_time))

    # # Now read the file back into a Python list object
    # with open('list_of_ret_list.txt', 'r') as f:
    #     a = json.loads(f.read())

    # E OF SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE

    bo.plot_convergence()

    # print("=" * 20)
    # print("Value of (a,b,c,d) that minimises the objective:" + str(bo.x_opt))
    # print("Minimum value of the objective: " + str(bo.fx_opt))
    # print("=" * 20)
    # print("=" * 30)
    # print('2nd part:')
    # print('This is the LOWEST NUMBER OF GENS:')
    # print(max(list_of_scores))
    #
    # print('this is the scores:')
    # print(list_of_scores)
    #
    # print(' this is the STD for ALL scores:')
    # print(np.std(list_of_scores))
    # print('SD of Scores')
    # print(np.std(list_of_scores))
    # print('MEAN of Scores')
    # print(np.mean(list_of_scores))
    # print(' this is how many evaluations are run for BO:')
    # print(len(list_of_scores))
    # print("=" * 30)
    # print("=" * 20)


    # ==============================================================
    # END OF Trying MODULAR BO
    # ==============================================================



# # Plotting Results
# y_rs = np.maximum.accumulate(rs.cv_results_['mean_test_score'])
# y_bo = np.maximum.accumulate(-optimizer.Y).ravel()
#
# print(f'Baseline neg. MSE = {baseline:.2f}')
# print(f'Random search neg. MSE = {y_rs[-1]:.2f}')
# print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')
#
# plt.plot(y_rs, 'ro-', label='Random search')
# plt.plot(y_bo, 'bo-', label='Bayesian optimization')
# plt.xlabel('Iteration')
# plt.ylabel('Neg. MSE')
# plt.ylim(-5000, -3000)
# plt.title('Value of the best sampled CV score');
# plt.legend();