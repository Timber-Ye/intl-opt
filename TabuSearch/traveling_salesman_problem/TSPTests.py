# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 00:38
# @Author  : 
# @File    : TSPTests.py.py

import datetime
import random
import unittest
import numpy as np
import os

import taboo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Fitness:
    """
    fitness value related operations
    """
    def __init__(self, totalDistance):
        self.TotalDistance = totalDistance  # Objective function value
        self.fitness = 1000 / self.TotalDistance  # Fitness value

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return "{:<10.3f}".format(self.TotalDistance)

    def get_fitness_value(self):
        return self.fitness

    def get_total_distance(self):
        return self.TotalDistance

    def get_improv(self, improv):
        return Fitness(self.TotalDistance + improv)


class Neighbor:
    """
    Neighbor related operations
    """
    def __init__(self, pos_1, pos_2, eval):
        self.Pos_1 = pos_1
        self.Pos_2 = pos_2
        self.Eval = eval

    def __gt__(self, other):
        return self.Eval + self.Punish < other.Eval + other.Punish

    def __str__(self):
        return "{}-{} eval: {:0.2f}".format(self.Pos_1, self.Pos_2, self.Eval)

    def punish(self, punish):
        """
        give some punishment based on previous frequency
        :param punish: punish_value = freq_in_middle_period_list * ratio
        """
        self.Punish = punish


def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    length = len(candidate.Route)
    if length < 10:
        print("{:<15}\t{}\t{}\t{}\n".format(
            '-'.join(map(str, candidate.Route)),
            candidate.Fitness, candidate.Strategy.name,
            timeDiff))
    else:
        print("{:<15}...{:<15}\t{}\t{}\t{}\n".format(
            '-'.join(map(str, candidate.Route[:5])),
            '-'.join(map(str, candidate.Route[-5:])),
            candidate.Fitness, candidate.Strategy.name,
            timeDiff))


def create(city_num):
    """
    路径表示，随机生成一个编码
    :param city_num: 城市个数
    :return:
    """
    _route = list(range(1, city_num))
    random.shuffle(_route)
    _route = [0] + _route
    return _route


def get_distance(idToLocationLookup):
    """
    from given data generate adj_mat, which contains distance between any two cities.
    """
    _n= idToLocationLookup.shape[0]
    adj_mat = np.zeros([_n, _n])
    for i in range(_n):
        for j in range(i, _n):
            adj_mat[i][j] = np.linalg.norm(np.subtract(idToLocationLookup[i],
                                                       idToLocationLookup[j]))
            adj_mat[j][i] = adj_mat[i][j]
    return adj_mat


def get_fitness(route, _adj_mat):
    fitness = _adj_mat[route[-1], route[0]]
    for i in range(len(route)-1):
        fitness += _adj_mat[route[i], route[i+1]]
    return Fitness(round(fitness, 3))


def get_improve(l, r, route, adj_mat):
    """
    2-opt
    """
    # bef: [..., l-1, l, l+1, ..., r-1, r, r+1]
    # aft: [..., l-1, r, r-1, ..., l+1, l, r+1]
    city_num = len(route)

    delta = adj_mat[route[l - 1]][route[r]] + adj_mat[route[l]][route[(r + 1) % city_num]] - \
            adj_mat[route[l - 1]][route[l]] - adj_mat[route[r]][route[(r + 1) % city_num]]

    return delta


def get_neighbor(route):
    """
    randomly choose two positions from [1, N-1]
    """
    [l, r, *_] = np.random.choice(range(1, len(route)), size=2, replace=False)
    if r < l:
        l, r = r, l
    return l, r


def move2neighbor(_route, _fitness, neighbor):
    route = _route[:]
    pos_1, pos_2 = neighbor.Pos_1, neighbor.Pos_2
    route[pos_1:pos_2+1] = route[pos_1:pos_2+1][::-1]
    # route[pos_1], route[pos_2] = route[pos_2], route[pos_1]
    return route, _fitness.get_improv(neighbor.Eval)


class TSPTests(unittest.TestCase):
    def solve(self, idToLocationLookup, optimalWeights=None,
              tabu_period=7, neighbor_range=None, poolSize=1, freq_punish=1e-3,
              generation=None):
        if neighbor_range is None:
            neighbor_range = [100, 200]
        startTime = datetime.datetime.now()
        cities_num = idToLocationLookup.shape[0]
        adj_mat = get_distance(idToLocationLookup)

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnCreate():
            return create(cities_num)

        def fnGetFitness(route):
            return get_fitness(route, adj_mat)

        def fnGetNeighbor(route):
            l, r = get_neighbor(route)
            _improv = get_improve(l, r, route, adj_mat)
            return Neighbor(l, r, _improv)

        def fnMove(route, fitness, neighbor):
            return move2neighbor(route, fitness, neighbor)

        optimalFitness = Fitness(optimalWeights)
        best, generation_mean_fitness, historical_best_fitness = taboo.get_best(cities_num, optimalFitness,
                                                get_fitness=fnGetFitness, display=fnDisplay,
                                                custom_generate=fnCreate, custom_get_neighbor=fnGetNeighbor,
                                                custom_move=fnMove,
                                                tabu_period=tabu_period, neighbor_range=neighbor_range,
                                                poolSize=poolSize, freq_punish=freq_punish,
                                                generation=generation)
        print("Optimal Solution: {}".format('-'.join(map(str, best.Route))))
        return best, generation_mean_fitness, historical_best_fitness


    def test_30_cities(self):
        """
        test case one
        :return:
        """
        idToLocationLookup = np.array([
            [41, 94],
            [37, 84],
            [54, 67],
            [25, 62],
            [7, 64],
            [2, 99],
            [68, 58],
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
        taboo.Benchmark.run(lambda tabu_length=15, neighbor_range=[100, 100],
                                   pool_size=1, freq_punish=4e-1:
                            self.solve(idToLocationLookup, opt_weights,
                                       tabu_period=tabu_length,
                                       neighbor_range=neighbor_range,
                                       poolSize=pool_size,
                                       generation=3000,
                                       freq_punish=freq_punish),
                            visualization=False)

    def test_38_cities(self):
        """
        test case two
        :return:
        """
        idToLocationLookup = np.loadtxt(BASE_DIR + '/test_38_cities.txt', delimiter=' ', usecols=[1, 2])
        idToLocationLookup = idToLocationLookup - np.amin(idToLocationLookup, axis=0)

        taboo.Benchmark.run(lambda tabu_length=20, neighbor_range=[100, 100],
                                   pool_size=3, freq_punish=5e-1:
                            self.solve(idToLocationLookup, 6657,
                                       tabu_period=tabu_length,
                                       neighbor_range=neighbor_range,
                                       poolSize=pool_size,
                                       generation=500,
                                       freq_punish=freq_punish),
                            visualization=True)


if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    unittest.main()


