# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/29 23:42
# @Author  : 
# @File    : particleSwarm.py

import math
import statistics
from itertools import count
from enum import Enum

import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def _generate():
    pass


def _custom_generate(custom_generate, get_fitness, _m):
    # while True:
    #     _p, _v = custom_generate(_m)
    #     fitness = get_fitness(_p, _m)
    #     if fitness.eval() != float('inf'):
    #         break
    _p, _v = custom_generate(_m)
    fitness = get_fitness(_p, _m)
    return Solution(_p, _v, fitness)


def _move(_p, _v, get_fitness, pbest_pos, gbest_pos, inertial=1.0, xi=1.0, c1=1.0, c2=1.0):
    pbest_delta = pbest_pos - _p
    gbest_delta = gbest_pos - _p
    # print("{} - {} = {}".format(pbest_pos.Xv, _p.Xv, pbest_delta.Xv))
    # print("{} - {} = {}".format(gbest_pos.Xv, _p.Xr, gbest_delta.Xv))
    new_v = _v.mov(pbest_delta, gbest_delta, inertial, c1, c2)
    new_p = _p + new_v
    fitness = get_fitness(new_p, 1)
    return Solution(new_p, new_v, fitness)


def _custom_move(custom_move, get_fitness):
    pass


def get_best(opt_fitness, get_fitness, display,
             custom_generate=None, custom_move=None,
             poolSize=50, inertial=1.0, c1=1.0, c2=1.0,
             xi=1.0, generation=None):
    historical_best_fitness = []
    generation_mean_fitness = []
    if custom_generate is None:
        def fnCreate():
            return _generate()
    else:
        def fnCreate(_m):
            return _custom_generate(custom_generate, get_fitness, _m)

    if custom_move is None:
        def fnMove(particle, pbest, gbest):
            return _move(particle.Position, particle.Velocity, get_fitness,
                         pbest.Position, gbest.Position,
                         inertial, xi, c1, c2)
    else:
        def fnMove(particle):
            pass

    strategyLookup = {
        Strategies.Init: lambda: fnCreate(0),
        Strategies.Create: lambda: fnCreate(1),
        Strategies.Move: lambda p, pbest, gbest: fnMove(p, pbest, gbest)
    }

    def _get_improvement(strategyLookup, pool_size, _generation):
        init = strategyLookup[Strategies.Init]()
        pbests = [init for _ in range(pool_size)]  # 初始化pbests
        pool = [strategyLookup[Strategies.Create]() for _ in range(pool_size)]  # 初始化粒子

        gbest = pbests[0]
        for _pbst in pbests:
            # display(_pbst)
            if _pbst.Fitness > gbest.Fitness:
                gbest = _pbst
        yield gbest

        iteration = count(0)

        while True:
            _w = [pool[i].Fitness.eval() for i in range(pool_size)]
            generation_mean_fitness.append(np.sum(_w) / pool_size)
            historical_best_fitness.append(gbest.Fitness.eval())

            iteration_num = next(iteration)
            if _generation is not None:
                if iteration_num > _generation - 1:
                    break

            for pool_idx in range(pool_size):
                new_particle = strategyLookup[Strategies.Move](pool[pool_idx], pbests[pool_idx], gbest)
                if new_particle.Fitness > pbests[pool_idx].Fitness:
                    pbests[pool_idx] = new_particle
                if new_particle.Fitness > gbest.Fitness:
                    gbest = new_particle
                    yield gbest

    improvement = None
    for improvement in _get_improvement(strategyLookup, poolSize, generation):
        display(improvement)
        if not opt_fitness > improvement.Fitness:
            historical_best_fitness.append(improvement.Fitness.eval())
            generation_mean_fitness.append(generation_mean_fitness[-1])
            break
    return improvement, generation_mean_fitness, historical_best_fitness


class Strategies(Enum):
    Init = 0,
    Create = 1,
    Move = 2


class Solution:
    def __init__(self, position, velocity, fitness):
        self.Position = position
        self.Velocity = velocity
        self.Fitness = fitness


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
            ax_1.legend(loc='upper right')
            # ax_2.legend(loc='upper right')
            fig.savefig('../../fig/[PSO]vehicle_routing_problem.pdf')
