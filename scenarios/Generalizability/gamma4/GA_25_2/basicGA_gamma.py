# Run a basic GA optimization on the Gamma scenario to find optimal actions
from pystorms.scenarios import gamma
import numpy as np
import pandas as pd
import random
import argparse
from deap import base
from deap import creator
from deap import tools


# Gamma Scenario for the basins in series
def GammaData(actions, CTRL_ASSETS=4):

    # Initialize the scenario
    env = gamma()
    done = False

    # Modify the logger function to store depths
    env.data_log["depthN"] = {}
    for i in np.linspace(1, CTRL_ASSETS, CTRL_ASSETS, dtype=int):
        env.data_log["depthN"][str(i)] = []

    # Dont log the params from upstream, we donot care about them
    for i in np.linspace(CTRL_ASSETS + 1, 11, 11 - CTRL_ASSETS, dtype=int):
        del env.data_log["flow"]["O" + str(i)]
        del env.data_log["flooding"][str(i)]

    # Simulate controlled response
    while not done:
        done = env.step(actions)

    # Return the logged params
    return env.data_log


# Objective Function
def f_loss(x):
    # GypOpt uses 2d array
    # pystorms requies 1d array

    # Simulate the response of control actions
    data = GammaData(x)

    # Convert to pandas dataframes
    depths = pd.DataFrame.from_dict(data["depthN"])
    flows = pd.DataFrame.from_dict(data["flow"])
    flooding = pd.DataFrame.from_dict(data["flooding"])

    # Compute loss - check the performance metric equation in Readme
    loss = 0.0

    # Flooding loss
    for node in flooding.keys():
        if flooding[node].sum() > 0.0:
            loss += 10 ** 4 * flooding[node].sum()

    # Flow deviation
    flows = flows.sub(4.0)
    flows[flows < 0.0] = 0.0
    loss += flows.sum().sum()

    # Prevent basins from storing water at the end.
    for i in depths.values[-1]:
        if i > 0.1:
            loss += i * 10 ** 3

    return loss


def initializeGA():
    # Attribute generator
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability)
    toolbox.register("attr_bool", random.random)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 4
    )

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized
    def evalOnePerfMin(individual):
        prfrmnce = f_loss(individual)
        return (prfrmnce,)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalOnePerfMin)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)


def findBestActionsGA(random_seed):
    random.seed(random_seed)
    # random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=15)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.3, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Total number of simulations
    sim_total = 0

    # Begin the evolution:
    while min(fits) > 0 and g < 4:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))
        sim_total += len(invalid_ind)

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Total evaluations: {} \n".format(sim_total))
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    data = {
        "Total_evaluations": sim_total,
        "Best_action": list(best_ind),
        "Minima": float(best_ind.fitness.values[0]),
    }
    return data


# Argument parser for batach run
parser = argparse.ArgumentParser(
    description="Generalizability Test GA Gamma4"
)
parser.add_argument("-seed", nargs=1, type=int, help="Random seed for the solver")
args = parser.parse_args()
# Read the parsed args
random_seed = args.seed[0]
# Set up GA
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

# Run GA
toolbox = base.Toolbox()
initializeGA()
data = findBestActionsGA(random_seed)
print(data)
name = str(random_seed) + "_GA_eval.npy"
np.save(name, data)
