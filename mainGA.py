from deap import base, creator, tools, algorithms
from sklearn.ensemble import GradientBoostingClassifier
from metrics import getKx2CVScores
import numpy as np
import random

# Load dataset
from myio import getLabelsFromCSVFile, getDatasetFromCSVFile
labels = getLabelsFromCSVFile('dataset/train_answers.csv')
[X_, y] = getDatasetFromCSVFile('dataset/train.csv', labels)
X = X_[:, 4:]

clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)

# Define the evaluation function
def eval_func(chromosome):
    features = [index for index, value in enumerate(chromosome) if value == 1]
    if not features:  # Handle empty feature sets
        return 0.0,
    scores = getKx2CVScores(clf, X[:, features], y, k=1)
    return np.mean(scores[0]),

# Define DEAP components
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)  # Binary attributes (0 or 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic algorithm parameters
population = toolbox.population(n=25)
NGEN = 20
CXPB = 0.5  # Crossover probability
MUTPB = 0.2  # Mutation probability

# Run the Genetic Algorithm
for gen in range(NGEN):
    print(f"Generation {gen}")
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fitnesses = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    population = toolbox.select(offspring, len(population))

# Get the best individual
best_ind = tools.selBest(population, k=1)[0]
print("Best individual is:", best_ind)
print("Fitness:", best_ind.fitness.values[0])

# Save the best chromosome
np.save('nGB', np.array(best_ind))
