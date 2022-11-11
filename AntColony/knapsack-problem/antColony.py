# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 15:03
# @Author  : Hanchiao
# @File    : antColony.py
import numpy as np
from enum import Enum
from itertools import count
import statistics

import matplotlib.pyplot as plt
import sys
import time


def _custom_create(custom_generate, pool):
    initial_loc, accessible = custom_generate()
    return Solution([initial_loc], pool, _access=accessible)


def _custom_route(ant, selectMap, custom_routing, get_fitness):
    _pool = ant.Pool
    _route = ant.Route[:]
    _access = ant.Access[:]
    while _access:
        _route, _access = custom_routing(_route, _access, selectMap.prob_query(_route[-1], _access))
    # print(_route)
    return Solution(_route, _pool, fitness=get_fitness(_route))


def get_best(items_num, opt_fitness,
             get_fitness, display,
             custom_generate=None, custom_move=None,
             forget_rate=0.5, alpha=1, beta=1, poolSize=10,
             generation=None):
    historical_best_fitness = []
    generation_mean_fitness = []
    select_map = SelectProbMap(items_num, get_fitness, _forget=forget_rate, _alpha=alpha, _beta=beta)

    if custom_generate is None:
        def fnCreate(pool):
            pass
    else:
        def fnCreate(pool):
            return _custom_create(custom_generate, pool)

    if custom_move is None:
        def fnMove(candidate):
            pass
    else:
        def fnMove(candidate):
            return _custom_route(candidate, select_map, custom_move, get_fitness)

    strategyLookup = {
        Strategies.Create: lambda pool=1: fnCreate(pool),
        Strategies.Explore: lambda candidate: fnMove(candidate)
    }

    def _get_improvement(strategyLookup, poolSize, iteration=None):
        pool = [strategyLookup[Strategies.Create](i) for i in range(poolSize)]  # 蚁群初始化
        _generation = count(0)

        routes = [strategyLookup[Strategies.Explore](pool[pool_index]) for pool_index in range(poolSize)]  # 探索一条完整解
        best = max(routes)
        generation_mean_fitness.append(best.Fitness.eval())
        historical_best_fitness.append(best.Fitness.eval())
        yield best

        while True:
            generation_num = next(_generation)
            if iteration is not None:
                if generation_num > iteration - 1:
                    break

            routes = [strategyLookup[Strategies.Explore](pool[pool_index]) for pool_index in range(poolSize)]
            local_best = max(routes)
            if best < local_best:
                best = local_best
                yield best

            _w = [routes[i].Fitness.eval() for i in range(poolSize)]
            generation_mean_fitness.append(np.sum(_w) / poolSize)
            historical_best_fitness.append(best.Fitness.eval())

            select_map.update(routes, best)
            # print("{}".format(select_map))

    improvement = None
    for improvement in _get_improvement(strategyLookup, poolSize, generation):
        display(improvement)
        if not opt_fitness > improvement.Fitness:
            historical_best_fitness.append(improvement.Fitness.eval())
            generation_mean_fitness.append(generation_mean_fitness[-1])
            break
    return improvement, generation_mean_fitness, historical_best_fitness


class Solution:
    def __init__(self, route, _pool, fitness=None, _access=None):
        self.Route = route[:]
        self.Fitness = fitness
        self.Pool = _pool
        self.Access = _access

    def __gt__(self, other):
        return self.Fitness > other.Fitness


class SelectProbMap:
    def __init__(self, _size, get_fitness, _forget, _alpha, _beta):
        self.Pheromone = np.ones([_size, _size])
        self.Size = _size
        self.Forget = _forget
        self.Alpha = _alpha
        self.Beta = _beta
        self.fnGetFitness = get_fitness

        self.Probability = np.zeros([_size, _size])
        for _i in range(_size):
            for _j in range(_size):
                self.Probability[_i, _j] = 1

    def prob_query(self, _i, _access):
        _sum = sum(self.Probability[_i, _access])
        probability = self.Probability[_i, _access] / _sum
        return probability

    def update(self, _routes, _best):
        for _i in range(self.Size):
            for _j in range(self.Size):
                if self.Pheromone[_i, _j] > 1e-20:
                    self.Pheromone[_i, _j] *= self.Forget  # 信息素衰减

        # for _r in _routes:
        #     _fit = self.fnGetFitness(_r.Route)
        #     for _i in range(len(_r.Route)-1):
        #         self.Pheromone[_r.Route[_i], _r.Route[_i+1]] += _fit.info()*1e-8
        #         self.Pheromone[_r.Route[_i+1], _r.Route[_i]] = self.Pheromone[_r.Route[_i], _r.Route[_i+1]]

        best_fit = self.fnGetFitness(_best.Route)  # 最优路径信息素增强
        for _i in range(len(_best.Route)-1):
            self.Pheromone[_best.Route[_i], _best.Route[_i+1]] += best_fit.info()
            self.Pheromone[_best.Route[_i+1], _best.Route[_i]] = self.Pheromone[_best.Route[_i], _best.Route[_i+1]]

        for _i in range(self.Size):  # 概率图更新
            for _j in range(_i):
                _fit = self.fnGetFitness([_j])
                _tmp = np.power(self.Pheromone[_i, _j], self.Alpha) * np.power(_fit.eval(), self.Beta)
                self.Probability[_i, _j] = _tmp
                self.Probability[_j, _i] = _tmp

    def __str__(self):
        return "Pheromone: {}".format(self.Pheromone)


class Strategies(Enum):
    Create = 0,
    Explore = 1


class Benchmark:
    @staticmethod
    def run(function, visualization=False):
        timings = []
        optimal_cost = []
        stdout = sys.stdout
        # print("\t{:3}\t{}\t{}".format("No.", "Mean", "Stdev"))
        # pool_size = [0, 5, 15, 50]
        # color = ['salmon', 'sandybrown', 'greenyellow', 'darkturquoise']

        # for i, value in enumerate(neighbor_range):
        for i in range(1):
            startTime = time.time()

            # sys.stdout = None  # avoid the output to be chatty
            best, generation_mean_fitness, historical_best_fitness = function()
            seconds = time.time() - startTime
            optimal_cost.append(best.Fitness.eval())
            sys.stdout = stdout

            if visualization:
                fig = plt.figure()
                ax_1 = fig.add_subplot(111)
                # ax_1.set_aspect(1)
                ax_1.set(ylabel='Collecting value')
                plt.grid(linestyle='--', linewidth=1, alpha=0.3)

                # ax_2 = fig.add_subplot(212)
                # # ax_2.set_aspect(1.2)
                # ax_2.set(xlabel='No. of generation', ylabel='Best so far')
                plt.grid(linestyle='--', linewidth=1, alpha=0.3)
                # fig.tight_layout()
                fig.suptitle('0-1 Bag Problem: Ant Colony Optimization', fontweight="bold")

                x_axis = len(generation_mean_fitness)
                x_axis = list(range(x_axis))

                ax_1.plot(x_axis, generation_mean_fitness, color='b', label="mean value", ls='-')
                ax_1.plot(x_axis, historical_best_fitness, color='g', label="best so far", ls='-')
                # ax_1.plot(x_axis, generation_mean_fitness, color=color[i], label='[{}, {}]'.format(value[0], value[1]), ls='-')
                # ax_2.plot(x_axis, historical_best_fitness, color=color[i], label='[{}, {}]'.format(value[0], value[1]), ls='-')

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
            fig.savefig('../../fig/[tmp-ACO]0-1 Bag Problem.pdf')
