# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/5 16:41
# @Author  : 
# @File    : TSPTests.py

import datetime
import random
import unittest
import math

import numpy as np

import genetic


class Fitness:
    def __init__(self, totalDistance):
        self.TotalDistance = totalDistance  # Objective function value
        self.fitness = 1000 / totalDistance  # Fitness value

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return "{:0.3f}".format(self.TotalDistance)

    def get_fitness_value(self):
        return self.fitness

    def get_total_distance(self):
        return self.TotalDistance


def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    length = len(candidate.Genes)
    if length < 10:
        print("{}\t{}\t{:<10}\t{}\n".format(
            '-'.join(map(str, candidate.Genes)),
            candidate.Fitness,
            candidate.Strategy.name,
            timeDiff))
    else:
        print("{}...{}\t{}\t{:<10}\t{}\n".format(
            '-'.join(map(str, candidate.Genes[:5])),
            '-'.join(map(str, candidate.Genes[-5:])),
            candidate.Fitness,
            candidate.Strategy.name,
            timeDiff))


def create(gene_set):
    """
    路径表示，随机生成一个编码
    :param gene_set: 编码取值范围
    :return:
    """
    _gene = gene_set[:]
    random.shuffle(_gene)  # 随机排列
    return _gene


def get_distance(location_a, location_b):
    """
    计算两个城市之间的距离
    :param location_a:
    :param location_b:
    :return:
    """
    side1 = location_a[0] - location_b[0]
    side2 = location_a[1] - location_b[1]
    side3 = math.sqrt(side1**2+side2**2)
    return side3



def get_fitness(genes, idToLocationLookup):
    """
    计算周游总路程
    :param genes: 周游序列编码
    :param idToLocationLookup: 城市坐标查询表
    :return: 总路程
    """
    fitness = get_distance(idToLocationLookup[genes[0]-1],
                           idToLocationLookup[genes[-1]-1])
    for i in range(len(genes)-1):
        fitness += get_distance(idToLocationLookup[genes[i]-1],
                                idToLocationLookup[genes[i+1]-1])
    return Fitness(round(fitness, 3))


def crossover(parentGenes, donorGenes):
    """
    次序杂交
    """
    length = len(parentGenes)
    [p1, p2, *_] = np.random.choice(range(1, length), size=2, replace=False)
    if p1 < p2:
        cut_1, cut_2 = p1, p2
    else:
        cut_1, cut_2 = p2, p1

    midSeg_p, midSeg_d = parentGenes[cut_1:cut_2], donorGenes[cut_1:cut_2]
    _p = parentGenes[cut_2:] + parentGenes[0:cut_1] + midSeg_p
    _d = donorGenes[cut_2:] + donorGenes[0:cut_1] + midSeg_d
    _p = [x for x in _p if not x in midSeg_d]
    _d = [x for x in _d if not x in midSeg_p]

    child_1 = _d[length-cut_2:] + midSeg_p + _d[0:length-cut_2]
    child_2 = _p[length-cut_2:] + midSeg_d + _p[0:length-cut_2]
    return child_1, child_2


def mutate(genes, mutate_rate):
    """
    互换变异
    :param genes: 编码
    :param mutate_rate: 变异率
    """
    childGenes = genes[:]

    '''Two different definitions of MUTATE_RATE'''

    # if random.uniform(0, 1) < mutate_rate:
    #     [l, r, *_] = np.random.choice(len(childGenes), size=2, replace=False)
    #     childGenes[l], childGenes[r] = childGenes[r], childGenes[l]
    #
    # return childGenes

    if random.uniform(0, 1) < mutate_rate * len(childGenes):
        [l, r, *_] = np.random.choice(len(childGenes), size=2, replace=False)
        childGenes[l], childGenes[r] = childGenes[r], childGenes[l]

    return childGenes
    # return childGenes


class TSPTests(unittest.TestCase):

    def solve(self, idToLocationLookup, optimalWeights=None,
              mutate_rate=0.01, crossover_rate=0.9,
              generation=None, poolSize=20):
        startTime = datetime.datetime.now()
        cities_num = idToLocationLookup.shape[0]
        gene_set = list(range(1, cities_num+1))

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnCreate():
            return create(gene_set)

        def fnGetFitness(genes):
            return get_fitness(genes, idToLocationLookup)

        def fnMutate(genes):
            return mutate(genes, mutate_rate)

        def fnCrossover(parentGenes, donorGenes):
            return crossover(parentGenes, donorGenes)

        optimalFitness = Fitness(optimalWeights)
        best, generation_mean_fitness, historical_best_fitness = genetic.get_best(cities_num, optimalFitness,
                                get_fitness=fnGetFitness, display=fnDisplay,
                                custom_crossover=fnCrossover, custom_mutate=fnMutate,
                                custom_create=fnCreate, poolSize=poolSize,
                                crossover_rate=crossover_rate, generation=generation)
        # self.assertTrue(best is not None)
        # self.assertFalse(best.Fitness < optimalFitness)

        print("Optimal Solution: {}".format('-'.join(map(str, best.Genes))))
        return best, generation_mean_fitness, historical_best_fitness

    # def test_benchmark(self):
    #     genetic.Benchmark.run(lambda: self.solve(length=30, mutate_rate=0.01,
    #                                              crossover_rate=0.9, generation=None,
    #                                              poolSize=20))

    def test_30_cities(self):
        idToLocationLookup = np.array([
            [41, 94],
            [37, 84],
            [54, 67],
            [25, 62],
            [7,64],
            [2, 99],
            [68,  58],
            [71, 44],
            [54, 62],
            [83, 69],
            [64, 60],
            [18, 54],
            [22, 60],
            [83, 46],
            [91, 38],
            [25, 38],
            [24, 42],
            [58, 69],
            [71, 71],
            [74, 78],
            [87, 76],
            [18, 40],
            [13, 40],
            [82, 7],
            [62, 32],
            [58, 35],
            [45, 21],
            [41, 26],
            [44, 35],
            [4, 50]
        ])
        opt_weights = 423.741
        # opt_weights = 500
        genetic.Benchmark.run(lambda mutate_rate=0.005, crossover_rate=0.9, poolSize=200:
                              self.solve(idToLocationLookup, opt_weights,
                                         mutate_rate=mutate_rate,
                                         crossover_rate=crossover_rate,
                                         generation=3000,
                                         poolSize=poolSize),
                              visualization=True)


if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    unittest.main()
