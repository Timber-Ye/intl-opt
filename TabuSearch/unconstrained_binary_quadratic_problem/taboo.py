# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 00:38
# @Author  : 
# @File    : taboo.py


import random
import statistics
from itertools import count
from enum import Enum
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def _generate(cities_num):
    pass


def _custom_generate(custom_generate, pool):
    route, fitness = custom_generate()
    return Solution(route, fitness, Strategies.Create, pool)


def _neighbor_search(current, get_neighbor):
    return get_neighbor(current.Route)


def _custom_move(solution, best_nb, strategy, move):
    route, fitness = move(solution.Route, solution.Fitness, best_nb)
    return Solution(route, fitness, strategy, solution.Pool)


def get_best(cities_num, opt_fitness, display,
             custom_generate=None, custom_get_neighbor=None, custom_move=None,
             tabu_period=7, poolSize=1, freq_punish=1e-3,
             neighbor_range=10, generation=None):
    historical_best_fitness = []
    generation_mean_fitness = []
    tabu_table = TabuTable(cities_num, tabu_period, freq_punish, poolSize)
    if custom_generate is None:
        def fnCreate(pool):
            return _generate(cities_num, pool)
    else:
        def fnCreate(pool):
            return _custom_generate(custom_generate, pool)

    if custom_get_neighbor is None:
        def fnGetNeighbor(route):
            pass
    else:
        def fnGetNeighbor(route):
            return _neighbor_search(route, custom_get_neighbor)

    if custom_move is None:
        def fnMove2Neighbor(solution, best_nb, strategy):
            pass
    else:
        def fnMove2Neighbor(solution, best_nb, strategy):
            return _custom_move(solution, best_nb, strategy, custom_move)


    strategyLookup = {
        Strategies.Create: lambda pool=1: fnCreate(pool),
        Strategies.Feasible: lambda s, n: fnMove2Neighbor(s, n, Strategies.Feasible),
        Strategies.BreakOut: lambda s, n: fnMove2Neighbor(s, n, Strategies.BreakOut)
    }

    def _get_improvement(strategyLookup, tabu_list, poolSize, neighbor_range, iteration=None):
        candidate_num = neighbor_range[0]
        neighbor_num = neighbor_range[1]
        local_bests = [strategyLookup[Strategies.Create](i) for i in range(poolSize)]
        pool = local_bests[:]
        global_best = local_bests[0]
        for _lb in local_bests:
            if _lb.Fitness > global_best.Fitness:
                global_best = _lb
        yield global_best

        generation = count(0)

        while True:
            _w = [local_bests[i].Fitness.get_total_distance() for i in range(poolSize)]
            generation_mean_fitness.append(np.sum(_w)/poolSize)
            historical_best_fitness.append(global_best.Fitness.get_total_distance())

            generation_num = next(generation)
            if iteration is not None:
                if generation_num > iteration - 1:
                    break

            for pool_index in range(poolSize):
                _nb = [fnGetNeighbor(pool[pool_index]) for _ in range(neighbor_num)]  # initialization
                for _n in _nb:
                    _n.punish(tabu_list.freq_punish(_n, pool_index))
                _nb.sort(reverse=True)
                for _index in range(candidate_num):
                    if tabu_list.tabu_check(_nb[_index], pool_index):  # if not tabu
                        pool[pool_index] = strategyLookup[Strategies.Feasible](pool[pool_index], _nb[_index])
                        tabu_list.add(_nb[_index], pool_index)
                        if pool[pool_index].Fitness > local_bests[pool_index].Fitness:
                            # if exceeds the best so far
                            local_bests[pool_index] = pool[pool_index]
                        break
                    else:  # if tabu
                        if local_bests[pool_index].Fitness < pool[pool_index].Fitness.get_improv(_nb[_index].Eval):
                            # breakout
                            pool[pool_index] = strategyLookup[Strategies.BreakOut](pool[pool_index], _nb[_index])
                            local_bests[pool_index] = pool[pool_index]
                            tabu_list.add(_nb[_index], pool_index)
                            break
                if local_bests[pool_index].Fitness > global_best.Fitness:  # update the best so far solution
                    global_best = local_bests[pool_index]
                    yield global_best
                tabu_list.update(pool_index)

    improvement = None
    for improvement in _get_improvement(strategyLookup, tabu_table, poolSize, neighbor_range, generation):
        display(improvement)
        if not opt_fitness > improvement.Fitness:
            historical_best_fitness.append(improvement.Fitness.get_total_distance())
            generation_mean_fitness.append(generation_mean_fitness[-1])
            break
    return improvement, generation_mean_fitness, historical_best_fitness


class Solution:
    def __init__(self, route, fitness, strategy, pool):
        self.Route = route
        self.Fitness = fitness
        self.Strategy = strategy
        self.Pool = pool

    def sol_2_str(self):
        idx = [i for i in range(len(self.Route)) if self.Route[i] == 1]
        return '-'.join(map(str, idx)), len(idx)


class Strategies(Enum):
    Create = 0,
    Feasible = 1,
    BreakOut = 2


class TabuTable:
    """
    two-dimensional Array Implementation for Tabu list, which includes short, middle, long period tenure.
    """
    def __init__(self, size, period, punish_rate=1e-3, poolSize=1):
        self.size = size
        self.tenure = [np.zeros(size) for _ in range(poolSize)]
        self.freq = [np.zeros(size) for _ in range(poolSize)]
        self.period = period + 1
        self.punish_rate = punish_rate

    def add(self, nb, pool=0):
        self.freq[pool][nb.Pos] += 1
        self.tenure[pool][nb.Pos] = self.period

    def update(self, pool=0):
        for _i in range(self.size):
            if self.tenure[pool][_i] != 0:
                self.tenure[pool][_i] -= 1

    def tabu_check(self, nb, pool=0):
        return self.tenure[pool][nb.Pos] == 0

    def freq_punish(self, nb, pool=0):
        return self.freq[pool][nb.Pos]*self.punish_rate

    def to_string(self, pool=0):
        return self.tenure[pool]


class Benchmark:
    @staticmethod
    def run(function, visualization=False):
        timings = []
        optimal_cost = []
        stdout = sys.stdout

        if visualization:
            fig = plt.figure()
            ax_1 = fig.add_subplot(111)
            # ax_1.set_aspect(1)
            ax_1.set(ylabel='$f(\mathbf{X})$', xlabel='No. of Generation')
            plt.grid(linestyle='--', linewidth=1, alpha=0.3)

        for i in range(100):
            startTime = time.time()

            sys.stdout = None  # avoid the output to be chatty
            best, generation_mean_fitness, historical_best_fitness = function()
            seconds = time.time() - startTime
            optimal_cost.append(best.Fitness.get_total_distance())
            sys.stdout = stdout

            if visualization:
                # np.save("tuning_neighbor_range/mean_fitness_{}.npy".format(i), np.array(generation_mean_fitness))
                # np.save("tuning_neighbor_range/best_fitness_{}.npy".format(i), np.array(historical_best_fitness))
                x_axis = len(generation_mean_fitness)
                x_axis = list(range(x_axis))

                if i == 0:
                    ax_1.plot(x_axis, generation_mean_fitness, color='b', alpha=0.3, label='mean cost')
                    ax_1.plot(x_axis, historical_best_fitness, color='g', alpha=0.35, label='best so far')
                else:
                    ax_1.plot(x_axis, generation_mean_fitness, color='b', alpha=0.1)
                    ax_1.plot(x_axis, historical_best_fitness, color='g', alpha=0.15)

            timings.append(seconds)
            mean_time = statistics.mean(timings)
            mean_cost = statistics.mean(optimal_cost)
            print("Time Consuming:\t{}\tOptimal Costs:\t{}".format(timings[i], optimal_cost[i]))

            # only display statistics for the first ten runs and then every 10th run after that.
            if i < 10 or i % 10 == 9:
                print("\t{:3}\tMean Time Consuming: {:<3.2f}\tStandard Deviation {:<3.2f}".format(
                    1 + i, mean_time,
                    statistics.stdev(timings, mean_time)
                    if i > 1 else 0))
                print("\t{:3}\tMean Traveling Cost: {:<3.2f}\tStandard Deviation {:<3.2f}".format(
                    1 + i, mean_cost,
                    statistics.stdev(optimal_cost, mean_cost)
                    if i > 1 else 0))

        if visualization:
            ax_1.legend(loc='lower right')
            # ax_2.legend(loc='upper right')
            fig.savefig('../../fig/[tmp]【TS】Unconstrained Binary Quadratic Problem.pdf')

        return best
