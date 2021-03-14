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
    generations = 150
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
random_list_of_ret_list=list()
def f_train_network_mean_num_gens(parameters): # returns number of generations until solved

    # print('This is parameters:  '+str(parameters))
    a = parameters[0][0]
    b = parameters[0][1]
    c = parameters[0][2]
    d = parameters[0][3]

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'gym_config')

    ret_list = list()

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

    random_list_of_ret_list.append(ret_list)
    sd = np.std(ret_list)
    score = np.mean(ret_list)/args.generations
    return [np.array(score),sd]

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

list_of_mins = list()
list_of_min_params = list()
list_of_min_variances = list()

#
# print("Input Nodes: %s" % str(len(my_env.observation_space.high)))
# print("Output Nodes: %s" % str(my_env.action_space.n))

if __name__ == '__main__':
    for j in range(1):

        bds = [{'name': 'prob_add_connection', 'type': 'continuous', 'domain': (0, 1)},
               {'name': 'prob_delete_connection', 'type': 'continuous', 'domain': (0, 1)},
               {'name': 'prob_add_node', 'type': 'continuous', 'domain': (0, 1)},
               {'name': 'prob_delete_node', 'type': 'continuous', 'domain': (0, 1)}]


        max_iter = 100
        list_of_scores = list()
        list_of_params = list()
        list_of_variances = list()
        a = 10

        for i in range(max_iter + a):
            parameters = np.array([[np.random.uniform(0, 1, 1), np.random.uniform(0, 1, 1), np.random.uniform(0, 1, 1), np.random.uniform(0, 1, 1)]])
            list_of_scores.append(f_train_network_mean_num_gens(parameters)[0])
            list_of_variances.append(f_train_network_mean_num_gens(parameters)[1])
            list_of_params.append(parameters)

        # SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE
        # SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE
        # SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE
        import json

        a = random_list_of_ret_list
        with open('random_list_of_ret_list.txt', 'w') as f:
            f.write(json.dumps(a))

        Random_time = [time.time() - start_time]

        # a = [1, 2, 3]
        with open('Random_time.txt', 'w') as f:
            f.write(json.dumps(Random_time))


        # # Now read the file back into a Python list object
        # with open('list_of_ret_list.txt', 'r') as f:
        #     a = json.loads(f.read())

        # E OF SAVE SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE  SAVE

        list_of_mins.append(min(list_of_scores))
        list_of_min_params.append(list_of_params[list_of_scores.index(min(list_of_scores))])
        list_of_min_variances.append(list_of_variances[list_of_scores.index(min(list_of_scores))])
        print('SD of Scores')
        print(np.std(list_of_scores))
        print('list_of_scores:')
        print(list_of_scores)
        print('MEAN of Scores')
        print(np.mean(list_of_scores))
        print('Scores')
        print(list_of_scores)


        # ==============================================================
        # END OF Trying MODULAR BO
    #     # ==============================================================
    # for j in range(1):
    #     print("=" * 20)
    #     # print(list_of_scores)
    #     print('Random Run: '+str(j))
    #     # print("Value of (a,b,c,d) that minimises the objective:" + str(bo.x_opt))
    #     # print("Minimum value of the objective: " + str(bo.fx_opt))
    #
    #     print("min: " + str(list_of_mins[j]))
    #     # print('this is the index of the min: ')
    #     # print(list_of_scores.index(min(list_of_scores)))
    #     print('this is the PARAMS of the min: ')
    #     print(list_of_min_params[j])
    #     print('this is the variance of the min')
    #     print(list_of_min_variances[j])
    #     print("=" * 20)




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