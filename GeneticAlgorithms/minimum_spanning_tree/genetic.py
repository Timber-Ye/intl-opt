# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 16:15
# @Author  : Hanchiao
# @File    : genetic.py

import random
import statistics
from itertools import count
from enum import Enum
import sys
import time

import numpy as np


def _generate(node_num, gene_express, get_fitness):
    pass
    genes = []
    genes.extend(random.choices(list(range(node_num)), k=node_num))

    fitness = get_fitness(genes)
    phenotype = gene_express(genes)
    return Chromosome(genes, fitness, Strategies.Create, phenotype)


def _custom_generate(custom_generate, gene_express, get_fitness):
    gene = custom_generate()
    phenotype = gene_express(gene)
    fitness = get_fitness(phenotype)
    return Chromosome(gene, fitness, Strategies.Create, phenotype)


def _mutate(parent, node_num, get_fitness):
    pass


def _mutate_custom(parent, custom_mutate, gene_express, get_fitness):
    childGenes = custom_mutate(parent.Genes)
    phenotype = gene_express(childGenes)
    fitness = get_fitness(phenotype)
    return Chromosome(childGenes, fitness, Strategies.Mutate, phenotype)



def _crossover(parent_1, parent_2):
    pass


def _crossover_custom(parent_1, parent_2, crossover, gene_express, get_fitness):
    childGenes = crossover(parent_1.Genes, parent_2.Genes)
    phenotype = gene_express(childGenes)
    fitness = get_fitness(phenotype)
    return Chromosome(childGenes, fitness, Strategies.Crossover, phenotype)


def get_best(node_num, opt_fitness, gene_express, get_fitness,
             display, custom_mutate=None, custom_crossover=None, custom_create=None,
             poolSize=1, crossover_rate=0.9,
             generation=None):
    if custom_mutate is None:
        def fnMutate(parent):
            return _mutate(parent, node_num, get_fitness)
    else:
        def fnMutate(parent):
            return _mutate_custom(parent, custom_mutate, gene_express, get_fitness)

    if custom_crossover is None:
        def fnCrossover(parent_1, parent_2):
            return _crossover(parent_1, parent_2)
    else:
        def fnCrossover(parent_1, parent_2):
            return _crossover_custom(parent_1, parent_2, custom_crossover, gene_express, get_fitness)

    if custom_create is None:
        def fnGenerateParent():
            return _generate(node_num, gene_express, get_fitness)
    else:
        def fnGenerateParent():
            return _custom_generate(custom_create, gene_express, get_fitness)

    strategyLookup = {
        Strategies.Create: lambda: fnGenerateParent(),
        Strategies.Mutate: lambda p: fnMutate(p),
        Strategies.Crossover: lambda p1, p2: fnCrossover(p1, p2)
    }
    historical_best_fitness = []
    generation_mean_fitness = []

    def _get_improvement(strategyLookup, poolSize, crossover_rate, iteration):
        best = strategyLookup[Strategies.Create]()
        population = [best]

        for _ in range(poolSize - 1):
            indiv = strategyLookup[Strategies.Create]()
            if indiv.Fitness > best.Fitness:
                best = indiv
            population.append(indiv)
        yield best
        generation = count(0)

        while True:
            generation_num = next(generation)
            if iteration is not None:
                if generation_num > iteration - 1:
                    break

            w = [population[i].Fitness for i in range(poolSize)]
            generation_mean_fitness.append(np.sum(w)/poolSize)
            historical_best_fitness.append(best.Fitness)

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

                # mutation
                child = strategyLookup[Strategies.Mutate](population[i])
                if child.Fitness > best.Fitness:
                    best = child
                    yield best
                population[i] = child

            # update population(selection)
            size = len(population)
            w = [population[i].Fitness for i in range(size)]
            w_min = min(w)
            w = [w[i] - w_min + 1 for i in range(size)]  # scaling

            population = random.choices(population, weights=w, k=poolSize)

    for improvement in _get_improvement(strategyLookup, poolSize,
                                        crossover_rate, generation):
        display(improvement)
        if not opt_fitness > improvement.Fitness:
            historical_best_fitness.append(improvement.Fitness)
            generation_mean_fitness.append(generation_mean_fitness[-1])
            return improvement, generation_mean_fitness, historical_best_fitness

    return None, generation_mean_fitness, historical_best_fitness


class Chromosome:
    def __init__(self, genes, fitness, strategy, phenotype):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy = strategy
        self.Phenotype = phenotype


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
        # print("\t{:3}\t{}\t{}".format("No.", "Mean", "Stdev"))
        for i in range(1):
            startTime = time.time()

            # sys.stdout = None  # avoid the output to be chatty
            function()
            seconds = time.time() - startTime
            # sys.stdout = stdout

            timings.append(seconds)
            mean = statistics.mean(timings)

            # only display statistics for the first ten runs and then every 10th run after that.
            if i < 10 or i % 10 == 9:
                print("\t{:3}\t{:<3.2f}\t{:<3.2f}".format(
                    1 + i, mean,
                    statistics.stdev(timings, mean)
                    if i > 1 else 0))

