# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/5 16:41
# @Author  : 
# @File    : genetic.py


import random
import statistics
from itertools import count
from enum import Enum
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def _generate(node_num, gene_express, get_fitness):
    pass
    genes = []
    genes.extend(random.choices(list(range(node_num)), k=node_num))
    fitness = get_fitness(genes)
    phenotype = gene_express(genes)
    return Chromosome(genes, fitness, Strategies.Create)


def _custom_generate(custom_generate, get_fitness):
    gene = custom_generate()
    fitness = get_fitness(gene)
    return Chromosome(gene, fitness, Strategies.Create)


def _mutate(parent, node_num, get_fitness):
    pass


def _mutate_custom(parent, custom_mutate, get_fitness):
    childGenes = custom_mutate(parent.Genes)
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Mutate)


def _crossover(parent_1, parent_2):
    pass


def _crossover_custom(parent_1, parent_2, crossover, get_fitness):
    childGenes_1, childGenes_2 = crossover(parent_1.Genes, parent_2.Genes)
    fitness_1 = get_fitness(childGenes_1)
    fitness_2 = get_fitness(childGenes_2)
    Chromo_1 = Chromosome(childGenes_1, fitness_1, Strategies.Crossover)
    Chromo_2 = Chromosome(childGenes_2, fitness_2, Strategies.Crossover)
    if not fitness_1 > fitness_2:
        return Chromo_2, Chromo_1
    else:
        return Chromo_1, Chromo_2


def get_best(cities_num, opt_fitness, get_fitness,
             display, custom_mutate=None, custom_crossover=None, custom_create=None,
             poolSize=1, crossover_rate=0.9,
             generation=None):
    if custom_mutate is None:
        def fnMutate(parent):
            return _mutate(parent, cities_num, get_fitness)
    else:
        def fnMutate(parent):
            return _mutate_custom(parent, custom_mutate, get_fitness)

    if custom_crossover is None:
        def fnCrossover(parent_1, parent_2):
            return _crossover(parent_1, parent_2)
    else:
        def fnCrossover(parent_1, parent_2):
            return _crossover_custom(parent_1, parent_2, custom_crossover, get_fitness)

    if custom_create is None:
        def fnGenerateParent():
            return _generate(cities_num, get_fitness)
    else:
        def fnGenerateParent():
            return _custom_generate(custom_create, get_fitness)

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

            w = [population[i].Fitness.get_fitness_value() for i in range(poolSize)]
            _w = [population[i].Fitness.get_total_distance() for i in range(poolSize)]
            generation_mean_fitness.append(np.sum(_w)/poolSize)
            historical_best_fitness.append(best.Fitness.get_total_distance())

            # w_min = min(w)
            # w = [w[i] - w_min + 0.05 for i in range(poolSize)]  # scaling
            # w_average = np.sum(w)/poolSize
            # w = [w[i] / w_average for i in range(poolSize)]

            for i in range(poolSize):
                # crossover
                if random.uniform(0, 1) < crossover_rate:
                    while True:
                        [p1, p2, *_] = random.choices(list(range(poolSize)), weights=w, k=2)  # roulette wheel selection
                        if not p1 == p2:
                            break
                    child_1, child_2 = strategyLookup[Strategies.Crossover](population[p1], population[p2])
                    population.extend([child_1, child_2])
                    if child_1.Fitness > best.Fitness:
                        best = child_1
                        yield best

                # mutation
                child = strategyLookup[Strategies.Mutate](population[i])
                if child.Fitness > best.Fitness:
                    best = child
                    yield best
                population[i] = child

            # update population(selection)
            size = len(population)
            w = [population[i].Fitness.get_fitness_value() for i in range(size)]
            w_min = min(w)
            w = [w[i] - w_min + 0.01 for i in range(size)]  # scaling

            population = random.choices(population, weights=w, k=poolSize)

    improvement = None
    for improvement in _get_improvement(strategyLookup, poolSize,
                                        crossover_rate, generation):
        display(improvement)
        if not opt_fitness > improvement.Fitness:
            historical_best_fitness.append(improvement.Fitness.get_total_distance())
            generation_mean_fitness.append(generation_mean_fitness[-1])
            return improvement, generation_mean_fitness, historical_best_fitness

    return improvement, generation_mean_fitness, historical_best_fitness


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
    def run(function, visualization=False):
        timings = []
        optimal_cost = []
        stdout = sys.stdout
        # print("\t{:3}\t{}\t{}".format("No.", "Mean", "Stdev"))
        # mutate_rate = [0.001, 0.005, 0.01, 0.02]
        # mutate_rate = [5e-3]
        poolsize=[20, 50, 100, 200]
        color = ['salmon', 'sandybrown', 'greenyellow', 'darkturquoise']

        if visualization:
            fig = plt.figure()
            ax_1 = fig.add_subplot(211)
            # ax_1.set_aspect(1)
            ax_1.set(ylabel='Average traveling costs', xlim=[0, 3000])
            plt.grid(linestyle='--', linewidth=1, alpha=0.3)

            ax_2 = fig.add_subplot(212)
            # ax_2.set_aspect(1.2)
            ax_2.set(xlabel='No. of generation', ylabel='Best so far',
                     xlim=[0, 3000])
            plt.grid(linestyle='--', linewidth=1, alpha=0.3)
            # fig.tight_layout()
            fig.suptitle('Tuning POOLSIZE', fontweight="bold")

        for i, value in enumerate(poolsize):
            startTime = time.time()

            sys.stdout = None  # avoid the output to be chatty
            best, generation_mean_fitness, historical_best_fitness = function(poolSize=value)
            seconds = time.time() - startTime
            optimal_cost.append(best.Fitness.get_total_distance())
            sys.stdout = stdout

            # visualization
            if visualization:
                x_axis = len(generation_mean_fitness)
                x_axis = list(range(x_axis))
                ax_1.plot(x_axis, generation_mean_fitness, color=color[i], label='{}'.format(value), ls='-')
                ax_2.plot(x_axis, historical_best_fitness, color=color[i], label='{}'.format(value), ls='-')

            timings.append(seconds)
            # mean_time = statistics.mean(timings)
            # mean_cost = statistics.mean(optimal_cost)
            print("Time Consuming:\t{}\tOptimal Costs:\t{}".format(timings[i], optimal_cost[i]))

            # only display statistics for the first ten runs and then every 10th run after that.
            # if i < 10 or i % 10 == 9:
            #     print("\t{:3}\tMean Time Consuming: {:<3.2f}\tStandard Deviation {:<3.2f}".format(
            #         1 + i, mean_time,
            #         statistics.stdev(timings, mean_time)
            #         if i > 1 else 0))
            #     print("\t{:3}\tMean Traveling Cost: {:<3.2f}\tStandard Deviation {:<3.2f}".format(
            #         1 + i, mean_cost,
            #         statistics.stdev(optimal_cost, mean_cost)
            #         if i > 1 else 0))

        ax_1.legend(loc='upper right')
        ax_2.legend(loc='upper right')
        fig.savefig('../../fig/[tmp]Tuning POOLSIZE.pdf')