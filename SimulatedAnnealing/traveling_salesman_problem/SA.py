# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/20 14:22
# @Author  : 
# @File    : SA.py
import math
import statistics
from itertools import count
from enum import Enum

import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def _generate(cities_num, get_fitness):
    pass


def _custom_generate(custom_generate, get_fitness, temperature):
    route = custom_generate()
    fitness = get_fitness(route)
    return Solution(route, fitness, Strategies.Create, temperature)


def _neighbor_search(current, get_neighbor):
    return get_neighbor(current.Route)


def _custom_move(solution, best_nb, _strategy, move):
    route, fitness = move(solution.Route, solution.Fitness, best_nb)
    strategy = None
    if _strategy == Strategies.Uphill:
        strategy = Strategies.Uphill
    else:
        if solution.Strategy == Strategies.Uphill:
            strategy = Strategies.Transition
        else:
            strategy = Strategies.Downhill

    return Solution(route, fitness, strategy, solution.Temperature)


def get_best(cities_num, opt_fitness,
             get_fitness, display,
             custom_generate=None, custom_get_neighbor=None, custom_move=None,
             inital_temp=100, inner_loop=200, cooling_rate=0.95,
             halt_temp=0.1, pool_size=200):
    historical_best_fitness = []
    historical_current = []
    warm_up_lim = 1
    warm_up = 0

    if custom_generate is None:
        def fnCreate():
            return _generate(cities_num, get_fitness)
    else:
        def fnCreate(t):
            return _custom_generate(custom_generate, get_fitness, t)

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
            nonlocal warm_up
            if strategy == Strategies.Trapped:
                solution.Trapped += 1
                if solution.Trapped > inner_loop * 20 and solution.Lifted < 2 and warm_up < warm_up_lim:
                    solution.lift_temperature()
                    warm_up += 1
                return solution
            solution.reset_trapped()
            return _custom_move(solution, best_nb, strategy, custom_move)

    def fnMetropolis(_neighbor, _t):
        if _neighbor.Eval < 0:
            return Strategies.Downhill
        else:
            if math.exp(-_neighbor.Eval/_t) > np.random.random(1):
                return Strategies.Uphill
            else:
                return Strategies.Trapped


    strategyLookup = {
        Strategies.Create: lambda t: fnCreate(t),
        Strategies.Downhill: lambda s, n: fnMove2Neighbor(s, n, Strategies.Downhill),
        Strategies.Uphill: lambda s, n: fnMove2Neighbor(s, n, Strategies.Uphill),
        Strategies.Trapped: lambda s, n: fnMove2Neighbor(s, n, Strategies.Trapped)
    }

    def _get_improvement(_strategyLookup, pool_size, init_t, inn_loop, cooling, halt_t):
        best = _strategyLookup[Strategies.Create](init_t)
        # currents = [_strategyLookup[Strategies.Create](init_t) for _ in range(pool_size)]
        # best = currents[-1]
        # for _i in range(pool_size-1):
        #     if currents[_i].Fitness > best.Fitness:
        #         best = currents[_i]
        yield best

        current = best

        while current.Temperature > halt_t:
            for _ in range(inn_loop):
                _neighbor = fnGetNeighbor(current)
                current = strategyLookup[fnMetropolis(_neighbor, current.Temperature)](current, _neighbor)
                if current.Fitness > best.Fitness:
                    best = current
                    yield best

            historical_current.append(current)
            historical_best_fitness.append(best.Fitness.get_total_distance())
            current.Temperature *= cooling

    improvement = None
    for improvement in _get_improvement(strategyLookup, pool_size, inital_temp, inner_loop, cooling_rate, halt_temp):
        display(improvement)
        if not opt_fitness > improvement.Fitness:
            historical_best_fitness.append(improvement.Fitness.get_total_distance())
            historical_current.append(historical_current[-1])
            break
    return improvement, historical_current, historical_best_fitness


class Solution:
    def __init__(self, route, fitness, strategy, _tmp):
        self.Route = route
        self.Fitness = fitness
        self.Strategy = strategy
        self.Temperature = _tmp
        self.Trapped = 0
        self.Lifted = 0

    def lift_temperature(self):
        self.Temperature += 100
        self.Lifted += 1
        self.Trapped = 0

    def reset_trapped(self):
        self.Trapped = 0


class Strategies(Enum):
    Create = 0,
    Downhill = 1,
    Uphill = 2,
    Trapped = 3,
    Transition = 4


class Benchmark:
    @staticmethod
    def run(function, visualization=False):
        timings = []
        optimal_cost = []
        stdout = sys.stdout
        # print("\t{:3}\t{}\t{}".format("No.", "Mean", "Stdev"))
        # tabu_length = [0, 5, 15, 50]
        # neighbor_range = [[20, 50], [50, 100], [100, 200], [200, 200]]
        # color = ['salmon', 'sandybrown', 'greenyellow', 'darkturquoise']

        ax_1 = None
        if visualization:
            fig = plt.figure()
            ax_1 = fig.add_subplot(111)
            ax_1.set(ylabel='Traveling cost', xlabel='No. of generation')
            ax_2 = ax_1.twinx()
            ax_2.set(ylabel='Temperature')
            plt.grid(linestyle='--', linewidth=1, alpha=0.3)
            fig.suptitle('Simulated Annealing - Traveling Salesman Problem', fontweight="bold")

        # for i, value in enumerate(neighbor_range):
        for i in range(20):
            startTime = time.time()

            sys.stdout = None  # avoid the output to be chatty
            best, historical_current, historical_best_fitness = function()
            seconds = time.time() - startTime
            optimal_cost.append(best.Fitness.get_total_distance())
            sys.stdout = stdout

            if visualization:
                # np.save("tuning_neighbor_range/mean_fitness_{}.npy".format(i), np.array(generation_mean_fitness))
                # np.save("tuning_neighbor_range/best_fitness_{}.npy".format(i), np.array(historical_best_fitness))
                x_axis = len(historical_best_fitness)
                x_axis = list(range(x_axis))
                historical_current_fitness = [_c.Fitness.get_total_distance() for _c in historical_current]
                historical_temperature = [_c.Temperature for _c in historical_current]
                ax_1.plot(x_axis, historical_current_fitness, color='C0', label='current cost', ls='-')
                ax_2.plot(x_axis, historical_temperature, color='C1', label='Temperature', ls='--')
                ax_1.plot(x_axis, historical_best_fitness, color='C2', label='best so far', ls='-')

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
            ax_1.legend(loc='upper right')
            ax_2.legend(loc='lower left')
            fig.savefig('../../fig/[tmp]【SA】Traveling salesman problem - 38 cities.pdf')

