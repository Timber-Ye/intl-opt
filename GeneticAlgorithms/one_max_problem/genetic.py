# File: genetic.py
# from chapter 2 of _Genetic Algorithms with Python_
import random
import statistics
from itertools import count
from enum import Enum
import sys
import time


def _generate_parent(length, geneSet, get_fitness):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.choices(geneSet, k=sampleSize))

    fitness = get_fitness(genes)
    return Chromosome(genes, fitness, Strategies.Create)


def _mutate(parent, geneSet, get_fitness):
    index = random.randrange(0, len(parent.Genes))
    childGenes = parent.Genes[:]
    newGene, alternative = random.sample(geneSet, 2)
    childGenes[index] = newGene \
        if newGene != childGenes[index] \
        else alternative
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Mutate)


def _mutate_custom(parent, custom_mutate, get_fitness):
    childGenes = custom_mutate(parent.Genes)
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Mutate)


def _crossover(parent_1, parent_2, get_fitness, crossover):
    childGenes = crossover(parent_1.Genes, parent_2.Genes)
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Crossover)


def get_best(get_fitness, targetLen, optimalFitness, geneSet,
             display, custom_mutate=None, crossover=None, custom_create=None,
             poolSize=1, crossover_rate=0.9,
             generation=None):
    if custom_mutate is None:
        def fnMutate(parent):
            return _mutate(parent, geneSet, get_fitness)
    else:
        def fnMutate(parent):
            return _mutate_custom(parent, custom_mutate, get_fitness)

    if custom_create is None:
        def fnGenerateParent():
            return _generate_parent(targetLen, geneSet, get_fitness)
    else:
        def fnGenerateParent():
            genes = custom_create()
            return Chromosome(genes, get_fitness(genes), Strategies.Create)

    strategyLookup = {
        Strategies.Create: lambda: fnGenerateParent(),
        Strategies.Mutate: lambda p: fnMutate(p),
        Strategies.Crossover: lambda p1, p2:
        _crossover(p1, p2, get_fitness, crossover)
    }
    usedStrategies = []

    for improvement in _get_improvement(strategyLookup, poolSize,
                                        crossover_rate, generation):
        display(improvement)
        f = strategyLookup[improvement.Strategy]
        usedStrategies.append(f)
        if not optimalFitness > improvement.Fitness:
            return improvement


def _get_improvement(strategyLookup, poolSize, crossover_rate, iteration):
    best = strategyLookup[Strategies.Create]()
    yield best
    population = [best]
    historicalFitnesses = [best.Fitness]
    for _ in range(poolSize - 1):
        indiv = strategyLookup[Strategies.Create]()
        if indiv.Fitness > best.Fitness:
            best = indiv
            yield best
            historicalFitnesses.append(best.Fitness)

        population.append(indiv)

    generation = count(0)
    while True:
        if iteration is not None:
            if next(generation) > iteration - 1:
                break

        w = [population[i].Fitness for i in range(poolSize)]
        w_min = min(w)
        w = [w[i] - w_min + 1 for i in range(poolSize)]  # scaling

        for i in range(poolSize):
            # crossover
            if random.uniform(0, 1) < crossover_rate:
                while True:
                    [p1, p2, *_] = random.choices(list(range(poolSize)), weights=w, k=2)  # roulette wheel selection
                    if not p1 == p2:
                        break
                child = strategyLookup[Strategies.Crossover](population[p1], population[p2])
                population.append(child)
                if child.Fitness > best.Fitness:
                    best = child
                    yield best
                    historicalFitnesses.append(best.Fitness)

            # mutation
            child = strategyLookup[Strategies.Mutate](population[i])
            if child.Fitness > best.Fitness:
                best = child
                yield best
                historicalFitnesses.append(best.Fitness)
            population[i] = child

        # update population(selection)
        size = len(population)
        w = [population[i].Fitness for i in range(size)]
        w_min = min(w)
        w = [w[i] - w_min + 1 for i in range(size)]  # scaling

        population = random.choices(population, weights=w, k=poolSize)


class Chromosome:
    def __init__(self, genes, fitness, strategy):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy = strategy


class Strategies(Enum):
    Create = 0,
    Mutate = 1,
    Crossover = 2,
    Select = 3


class Benchmark:
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(50):
            startTime = time.time()

            sys.stdout = None  # avoid the output to be chatty
            function()
            seconds = time.time() - startTime
            sys.stdout = stdout

            timings.append(seconds)
            mean = statistics.mean(timings)

            # only display statistics for the first ten runs and then every 10th run after that.
            if i < 10 or i % 10 == 9:
                print("{} {:3.2f} {:3.2f}".format(
                    1 + i, mean,
                    statistics.stdev(timings, mean)
                    if i > 1 else 0))
