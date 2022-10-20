# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 00:38
# @Author  : 
# @File    : TSPTest.py.py

import datetime
import random
import unittest
import math

import numpy as np

import taboo


class Fitness:
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
    def __init__(self, pos_1, pos_2, eval):
        self.Pos_1 = pos_1
        self.Pos_2 = pos_2
        self.Eval = eval

    def __gt__(self, other):
        return self.Eval + self.Punish < other.Eval + other.Punish

    def __str__(self):
        return "{}-{} eval: {:0.2f}".format(self.Pos_1, self.Pos_2, self.Eval)

    def punish(self, punish):
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
    _route = list(range(city_num))
    random.shuffle(_route)  # 随机排列
    return _route


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


def get_fitness(route, idToLocationLookup):
    """
    计算周游总路程
    :param route: 周游序列编码
    :param idToLocationLookup: 城市坐标查询表
    :return: 总路程
    """
    fitness = get_distance(idToLocationLookup[route[-1]],
                           idToLocationLookup[route[0]])
    for i in range(len(route)-1):
        fitness += get_distance(idToLocationLookup[route[i]],
                                idToLocationLookup[route[i+1]])
    return Fitness(round(fitness, 3))


def get_improve(l, r, route, idToLocationLookup):
    # bef: [..., i-1, l, i+1, ..., j-1, r, j+1]
    # aft: [..., i-1, r, i+1, ..., j-1, l, j+1]
    city_num = len(route)
    if r - l == 1:
        _d = get_distance(idToLocationLookup[route[l-1]], idToLocationLookup[route[l]])\
              + get_distance(idToLocationLookup[route[r]], idToLocationLookup[route[(r+1) % city_num]])
        d = get_distance(idToLocationLookup[route[l-1]], idToLocationLookup[route[r]])\
              + get_distance(idToLocationLookup[route[l]], idToLocationLookup[route[(r+1) % city_num]])
        return d-_d
    else:
        _dl = get_distance(idToLocationLookup[route[l-1]],idToLocationLookup[route[l]])\
              + get_distance(idToLocationLookup[route[l]],idToLocationLookup[route[l+1]])
        _dr = get_distance(idToLocationLookup[route[r-1]],idToLocationLookup[route[r]])\
              + get_distance(idToLocationLookup[route[r]],idToLocationLookup[route[(r+1) % city_num]])
        dl = get_distance(idToLocationLookup[route[l-1]],idToLocationLookup[route[r]])\
              + get_distance(idToLocationLookup[route[r]],idToLocationLookup[route[l+1]])
        dr = get_distance(idToLocationLookup[route[r-1]],idToLocationLookup[route[l]])\
              + get_distance(idToLocationLookup[route[l]],idToLocationLookup[route[(r+1) % city_num]])

        return dl + dr - _dl - _dr

def get_neighbor(route):
    """

    :param route:
    :return:
    """
    [l, r, *_] = np.random.choice(range(1, len(route)), size=2, replace=False)
    np.random.sample()
    if r < l:
        l, r = r, l
    return l, r


def move2neighbor(_route, _fitness, neighbor):
    route = _route[:]
    pos_1, pos_2 = neighbor.Pos_1, neighbor.Pos_2
    route[pos_1], route[pos_2] = route[pos_2], route[pos_1]
    return route, _fitness.get_improv(neighbor.Eval)


class TSPTests(unittest.TestCase):
    def solve(self, idToLocationLookup, optimalWeights=None,
              tabu_period=7, neighbor_range=None, poolSize=1, freq_punish=1e-3,
              generation=None):
        if neighbor_range is None:
            neighbor_range = [100, 200]
        startTime = datetime.datetime.now()
        cities_num = idToLocationLookup.shape[0]

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnCreate():
            return create(cities_num)

        def fnGetFitness(route):
            return get_fitness(route, idToLocationLookup)

        def fnGetNeighbor(route):
            l, r = get_neighbor(route)
            _improv = get_improve(l, r, route, idToLocationLookup)
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
                                   pool_size=5, freq_punish=4e-1:
                            self.solve(idToLocationLookup, opt_weights,
                                       tabu_period=tabu_length,
                                       neighbor_range=neighbor_range,
                                       poolSize=pool_size,
                                       generation=3000,
                                       freq_punish=freq_punish),
                            visualization=False)


if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    unittest.main()


