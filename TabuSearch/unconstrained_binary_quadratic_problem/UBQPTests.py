# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 00:38
# @Author  : 
# @File    : TSPTests.py.py

import datetime
import random
import unittest
import math
import numpy as np
import os

import taboo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Fitness:
    """
    fitness value related operations
    """
    def __init__(self, _value):
        self.fitness = _value

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return "{:<10.0f}".format(self.fitness)

    def get_fitness_value(self):
        return self.fitness

    def get_total_distance(self):
        return self.fitness

    def get_improv(self, improv):
        return Fitness(self.fitness + improv)


class Neighbor:
    """
    Neighbor related operations
    """
    def __init__(self, pos, eval):
        self.Pos = pos
        self.Eval = eval

    def __gt__(self, other):
        return self.Eval - self.Punish > other.Eval - other.Punish

    def __str__(self):
        return "{} eval: {:0.2f}".format(self.Pos, self.Eval)

    def punish(self, punish):
        """
        give some punishment based on previous frequency
        :param punish: punish_value = freq_in_middle_period_list * ratio
        """
        self.Punish = punish


def display(candidate, startTime):
    """
    for better output
    """
    timeDiff = datetime.datetime.now() - startTime
    _sol, length = candidate.sol_2_str()
    if length < 10:
        print("{:<33}\t{}\t{}\t{}\n".format(
            _sol,
            candidate.Fitness, candidate.Strategy.name,
            timeDiff))
    else:
        print("{:<15}...{:>15}\t{}\t{}\t{}\n".format(
            _sol[:15],
            _sol[-15:],
            candidate.Fitness, candidate.Strategy.name,
            timeDiff))


def create(_size):
    """
    choose 0 as the start point, whose fitness value is easy to derive.
    """
    _sol = [0] * _size
    return _sol, Fitness(0)


def get_quadratic_matrix(_size, adj_mat):
    """
    generate quadratic matrix in the upper triangle form from the given data
    """
    _quad = np.zeros([_size, _size])
    for _i in adj_mat:
        _quad[_i[0]-1, _i[1]-1] = (2 * _i[2]) if _i[0] != _i[1] else _i[2]  # upper triangle
    return _quad


def get_improve(_l, _sol, _quad):
    """
    get the moving value
    """
    _size = len(_sol)
    _flag = 1 - 2 * _sol[_l]
    delta = _quad[_l, _l]  # q_ii
    for _i in range(0, _size):
        if _i == _l or _sol[_i] == 0:
            continue
        delta += (_quad[_i, _l] if _i < _l else _quad[_l, _i])  # delta += q_{(i, j)}
    return delta * _flag


def get_neighbor(_sol):
    return np.random.choice(range(len(_sol)), size=1)[0]


def move2neighbor(_sol, _fitness, neighbor):
    sol_ = _sol[:]
    sol_[neighbor.Pos] = 1 - sol_[neighbor.Pos]  # one-flip
    return sol_, _fitness.get_improv(neighbor.Eval)


class TSPTests(unittest.TestCase):
    def solve(self, _size, _lookup, optimalWeights=None,
              tabu_period=7, neighbor_range=None, poolSize=1, freq_punish=1e-3,
              generation=None):
        if neighbor_range is None:
            neighbor_range = [100, 200]
        startTime = datetime.datetime.now()
        quad = get_quadratic_matrix(_size, _lookup)

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnCreate():
            return create(_size)

        def fnGetNeighbor(sol):
            l = get_neighbor(sol)
            _improv = get_improve(l, sol, quad)
            return Neighbor(l, _improv)

        def fnMove(sol, fitness, neighbor):
            return move2neighbor(sol, fitness, neighbor)

        optimalFitness = Fitness(optimalWeights)
        best, generation_mean_fitness, historical_best_fitness = taboo.get_best(_size, optimalFitness,
                                                display=fnDisplay,
                                                custom_generate=fnCreate, custom_get_neighbor=fnGetNeighbor,
                                                custom_move=fnMove,
                                                tabu_period=tabu_period, neighbor_range=neighbor_range,
                                                poolSize=poolSize, freq_punish=freq_punish,
                                                generation=generation)
        print("Optimal Solution: {}".format(best.sol_2_str()))
        return best, generation_mean_fitness, historical_best_fitness

    def test_bqp50(self):
        """
        test case one: bqp50.1
        """
        size = 50
        lookup = np.loadtxt(BASE_DIR + '/bqp50.txt', dtype=int)
        taboo.Benchmark.run(lambda tabu_length=20, neighbor_range=[100, 100],
                                   pool_size=1, freq_punish=0:
                            self.solve(size, lookup, optimalWeights=30000,
                                       tabu_period=tabu_length,
                                       neighbor_range=neighbor_range,
                                       poolSize=pool_size,
                                       generation=500,
                                       freq_punish=freq_punish),
                            visualization=True)

    def test_bqp2500(self):
        """
        test case one: bqp2500.1
        """
        size = 2500
        lookup = np.loadtxt(BASE_DIR + '/bqp2500.txt', dtype=int)
        taboo.Benchmark.run(lambda tabu_length=20, neighbor_range=[100, 100],
                                   pool_size=1, freq_punish=0:
                            self.solve(size, lookup, optimalWeights=1515944,
                                       tabu_period=tabu_length,
                                       neighbor_range=neighbor_range,
                                       poolSize=pool_size,
                                       generation=1000,
                                       freq_punish=freq_punish),
                            visualization=False)

    def test_bqp250(self):
        """
        test case one: bqp250.1
        """
        size = 250
        lookup = np.loadtxt(BASE_DIR + '/bqp250.txt', delimiter=' ', dtype=int)
        opt_weights = 3120000
        taboo.Benchmark.run(lambda tabu_length=20, neighbor_range=[100, 100],
                                   pool_size=4, freq_punish=0:
                            self.solve(size, lookup, optimalWeights=opt_weights,
                                       tabu_period=tabu_length,
                                       neighbor_range=neighbor_range,
                                       poolSize=pool_size,
                                       generation=500,
                                       freq_punish=freq_punish),
                            visualization=True)


if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    unittest.main()


