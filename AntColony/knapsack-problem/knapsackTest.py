# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 15:03
# @Author  : 
# @File    : knapsackTest.py

import antColony as AC
import datetime
import random
import unittest

import numpy as np

class Fitness:
    def __init__(self, _value):
        self.fitness = _value  # Fitness value

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return "{:^7.1f}".format(self.fitness)

    def eval(self):
        return self.fitness

    def info(self):
        return self.fitness*1e-3


def get_next_step(_route, _volume, itemWeightsLookup, gene_set):  # 确定下一步可选集合
    _w = 0
    if len(_route) != 0:
        for _i in _route:
            _w += itemWeightsLookup[_i]

    _r = _volume - _w
    next_step = []
    for _i in gene_set:
        if itemWeightsLookup[_i] <= _r and _i not in _route:
            next_step.append(_i)
    return next_step


def get_fitness(_route, itemValueLookup):  # 计算适应值
    _value = 0
    for _i in _route:
        _value += itemValueLookup[_i]

    return Fitness(_value)


def create(gene_set):
    return np.random.choice(gene_set)


def one_step(pro):  # 确定下一步
    _idx = 0
    if len(pro) == 1:
        return _idx
    for _idx, _i in enumerate(pro[:-1]):
        if random.uniform(0, 1) < _i:
            return _idx
    return _idx+1


def move(_route, _access, _probability):
    idx = one_step(_probability)
    _route.append(_access[idx])
    return _route


def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    length = len(candidate.Route)
    if length < 10:
        print("{}\t{}\t{:^4}\t{}\n".format(
            '-'.join(map(str, candidate.Route)),
            candidate.Fitness,
            candidate.Pool,
            timeDiff))
    else:
        print("{}...{}\t{}\t{:^4}\t\t{}\n".format(
            '-'.join(map(str, candidate.Route[:5])),
            '-'.join(map(str, candidate.Route[-5:])),
            candidate.Fitness,
            candidate.Pool,
            timeDiff))


class KnapsackTests(unittest.TestCase):
    def solve(self, volume, itemWeightsLookup, itemValueLookup,
              optimalValue=None, forget_rate=0.5, alpha=1.0, beta=1.0,
              poolSize=20, generation=None):
        startTime = datetime.datetime.now()
        assert itemWeightsLookup.shape == itemValueLookup.shape
        item_num = itemWeightsLookup.shape[0]
        gene_set = get_next_step([], volume, itemWeightsLookup, list(range(item_num)))

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnCreate():
            start_point = create(gene_set)
            next_steps = get_next_step([start_point], volume, itemWeightsLookup, gene_set)
            return start_point, next_steps

        def fnMove(route, access, probability):
            _route = move(route, access, probability)
            next_steps = get_next_step(_route, volume, itemWeightsLookup, gene_set)
            return _route, next_steps

        def fnGetFitness(route):
            return get_fitness(route, itemValueLookup)

        optimalFitness = Fitness(optimalValue)
        best, generation_mean, historical_best_fitness = AC.get_best(item_num, optimalFitness, fnGetFitness,
                                                                     fnDisplay, custom_generate=fnCreate,
                                                                     custom_move=fnMove,
                                                                     forget_rate=forget_rate,
                                                                     alpha=alpha, beta=beta,
                                                                     poolSize=poolSize, generation=generation)
        print("Optimal Solution: {}".format('-'.join(map(str, best.Route))))
        return best, generation_mean, historical_best_fitness

    def test_10_items(self):
        itemValueLookup = np.array([55, 10, 47, 5, 4, 50, 8, 61, 85, 87])
        itemWeightsLookup = np.array([95, 4, 60, 32, 23, 72, 80, 62, 65, 46])
        volume = 269
        opt_value = 295
        AC.Benchmark.run(lambda alpha=0.5, beta=0.5,
                                forget_rate=0.8, pool_size=3,
                                generation=1000: self.solve(volume, itemWeightsLookup, itemValueLookup,
                                                            opt_value, alpha=alpha, beta=beta, forget_rate=forget_rate,
                                                            poolSize=pool_size, generation=generation),
                         visualization=True)

    # def test_100_items(self):
    #     data = np.loadtxt("test_case_100.txt", delimiter=' ')
    #     itemWeightsLookup = data[:, 0]
    #     itemValueLookup = data[:, 1]
    #     volume = 1000
    #     opt_value = 2614
    #     AC.Benchmark.run(lambda alpha=1.5, beta=15,
    #                             forget_rate=0.2, pool_size=100,
    #                             generation=200: self.solve(volume, itemWeightsLookup, itemValueLookup,
    #                                                         opt_value, alpha=alpha, beta=beta, forget_rate=forget_rate,
    #                                                         poolSize=pool_size, generation=generation),
    #                      visualization=True)


if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    unittest.main()

