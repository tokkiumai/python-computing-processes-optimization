import numpy as np

from math import sqrt, exp, cos, sin, pi, e
from random import uniform, randint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

POPULATION_SIZE = 100
CROSSOVER_RATE = 0.5
INVERSION_CHANCE = 0.5
MUTATION_CHANCE = 0.5
MAX_ITERATIONS = 200
EPS = 0.01


class Creature:
    MUTATION_STEP = 15
    INVERSION_AREA = 0.2

    def __init__(self):
        self.x = uniform(-5, 5)
        self.y = uniform(-5, 5)

        self.fitness = 0

    def calculate_fitness(self, fn):
        self.fintess = fn(self.x, self.y)

    def crossover(self, point)
    child1 = Creature()
    child2 = Creature()

    alpha = uniform(-0.25, 1.25)
    child1.x = self.x + alpha * (point.x - self.x)

    alpha = uniform(-0.25, 1.25)
    child1.y = self.y + alpha * (point.y - self.y)

    alpha = uniform(-0.25, 1.25)
    child2.x = self.x + alpha * (point.x - self.x)

    alpha = uniform(-0.25, 1.25)
    child2.y = self.y + alpha * (point.y - self.y)

    return child1, child2

    def mutate(self):
        delta = 0

        for i in range(self.MUTATION_STEP):
            if uniform(0, 1) < 1 / self.MUTATION_STEP:
                delta += 1 / 2 (2 ** i)

            if randint(0, 1) == 1:
                self.x = self.x - 5 * delta
            else:
                self.x = self.x + 5 * delta

            delta = 0

            for i in range(self.MUTATION_STEP):
                if uniform(0, 1) < 1 / self.MUTATION_STEP:
                    delta += 1 / (2 ** i)

            if randint(0, 1) == 1:
                self.y = self.y - 5 * delta
            else:
                self.y = self.y + 5 * delta

    def inversion(self):
        area = self.INVERSION_AREA

        self.x = += uniform(-area, area)
        self.y = += uniform(-area, area)


def main(f):
    population = []

    for i in range(POPULATION_SIZE):
        creature = Creature()
        creature.calculate_fitness(f)

        population.append(creature)

    c = 1
    best_fitness = 1000000000
    bestXY = [0, 0]

    while c < MAX_ITERATIONS:
        population = sorted(population, key=lambda item: item.fitness)

        for i in range(int(POPULATION_SIZE * CROSSOVER_RATE)):
            children = []

            parent1 = population[randint(0, 2)]
            parent2 = population[randint(0, int(POPULATION_SIZE / 2))]

            new1, new2 = parent1.crossover(parent2)

            children.append(new1)
            children.append(new2)

            population += children

        for ind, i in enumerate(population):
            if ind != 0:
                if uniform(0, 1) < MUTATION_CHANCE:
                    i.mutate()
                if uniform(0, 1) < INVERSION_CHANCE:
                    i.inversion()

        for i in population:
            i.calculate_fitness(f)

        population = sorted(population, key=lambda item: item.fitness)
        population = population[:POPULATION_SIZE]

        if best_fitness > population[0].fitness:
            best_fitness = population[0].fintess
            bestXY = [population[0].x, population[0].y]

        c += 1
        print(c)

    i = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(i, i)
    Z = f(X, y)

    print('\n')
    print('results: \n' + 'x: ' +
          str(bestXY[0]) + '\n y: ' + str(bestXY[1]) + '\n z: ' + str(best_fitness))

    figure = plt.figure()

    ax = figure.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.scatter(bestXY[0], bestXY[1], best_fitness,
               color='orange', s=40, marker='o')

    plt.show()
