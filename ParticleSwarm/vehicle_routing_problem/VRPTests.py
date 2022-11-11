# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/29 23:42
# @Author  : 
# @File    : VRPTests.py
import random
import unittest

import particleSwarm
import numpy as np
import math
import datetime


class Fitness:
    def __init__(self, totalDistance):
        self.TotalDistance = totalDistance
        self.Fitness = self.TotalDistance

    def __gt__(self, other):
        return self.Fitness < other.Fitness

    def __str__(self):
        return "{:^7.2f}".format(self.TotalDistance)

    def eval(self):
        return self.TotalDistance


class Velocity:
    def __init__(self, _v_v, _v_r):
        self.Vv = _v_v
        self.Vr = _v_r

    def mov(self, pd, gd, _inertial, c1, c2):  # update velocity
        indiv = pd * (c1 * np.random.rand())
        social = gd * (c2 * np.random.rand())
        new_Vv = indiv.Xv + social.Xv + _inertial * self.Vv
        new_Vr = indiv.Xr + social.Xr + _inertial * self.Vr
        return Velocity(new_Vv, new_Vr)

    def __mul__(self, other: float):
        return Velocity(self.Vv * other, self.Vr * other)


class Position:
    def __init__(self, _x_v, _x_r, _truck_num):
        self.Xv = _x_v
        self.Xr = _x_r
        self.Truck = _truck_num
        self.Dimension = len(_x_v)

    def __sub__(self, other):
        new_Xv = self.Xv - other.Xv
        new_Xr = self.Xr - other.Xr
        return Position(new_Xv, new_Xr, self.Truck)

    def __add__(self, other: Velocity):
        new_Xv = self.Xv + other.Vv
        new_Xr = self.Xr + other.Vr
        new_pos = Position(new_Xv, new_Xr, self.Truck)
        new_pos.validate()
        # print("new_p_Xv: {}\t new_p_Xr: {}".format(new_pos.Xv, new_pos.Xr))
        return new_pos

    def __mul__(self, other:float):
        new_Xv = other * self.Xv
        new_Xr = other * self.Xr
        return Position(new_Xv, new_Xr, self.Truck)

    def validate(self):  # normalization
        self.Xv = np.ceil(self.Xv)
        self.Xv = np.clip(self.Xv, 1, self.Truck)

    def __str__(self):
        str_ = ''
        for i in range(self.Truck):
            str_1 = '0-'
            depot = np.where(self.Xv == i+1)[0]
            arrange = self.Xr[depot].argsort()
            str_1 = str_1 + '-'.join(map(str, depot[arrange]+1))
            str_1 += '-0'
            str_ += '{:^15}'.format(str_1)
        return str_


def get_fitness(_p, truck_num, truck_limit, idToLocLookup, idToDemandsLookup):  # 计算适应值
    fit = 0
    for i in range(1, truck_num+1):
        depot = np.where(_p.Xv == i)[0]
        if len(depot) == 0:
            # print('[]')
            continue
        # print(depot+1)
        if idToDemandsLookup[depot].sum() > truck_limit:
            # print('inf')
            return float('inf')
        arrange = _p.Xr[depot].argsort()
        _p.Xr[depot] = arrange
        fit += get_distance(idToLocLookup[0], idToLocLookup[depot[arrange[0]]+1])
        for j in range(0, len(arrange) - 1):
            fit += get_distance(idToLocLookup[depot[arrange[j]]+1], idToLocLookup[depot[arrange[j+1]]+1])
        fit += get_distance(idToLocLookup[0], idToLocLookup[depot[arrange[-1]]+1])

    # print(fit)
    return fit


def create(truck_num, depot_num):
    Xv = np.random.choice(range(1, truck_num+1), depot_num, replace=True)
    Xr = np.random.uniform(1, depot_num, depot_num)

    Vv = np.random.choice(range(-truck_num+1, truck_num), depot_num, replace=True)
    Vr = np.random.uniform(-depot_num+1, depot_num-1, depot_num)

    return Position(Xv, Xr, truck_num), Velocity(Vv, Vr)


def get_distance(location_a, location_b):
    """
    计算两个城市之间的距离
    :param location_a:
    :param location_b:
    :return:
    """
    side1 = location_a[0] - location_b[0]
    side2 = location_a[1] - location_b[1]
    side3 = math.sqrt(side1 ** 2 + side2 ** 2)
    return side3


def display(candidate, startTime):  # 打印输出
    timeDiff = datetime.datetime.now() - startTime
    route = '{}'.format(candidate.Position)
    if len(route) < 100:
        print("{:<15}\t{}\t{}".format(route,
                                      candidate.Fitness,
                                      timeDiff))
    else:
        print("{:<15}...{:<15}\t{}\t{}".format(
            route[:10], route[-10:], candidate.Fitness, timeDiff
        ))


class VRPTests(unittest.TestCase):
    def solve(self, truck_num, truck_capacity, idToLocLookup, idToDemandsLookup,
              optimalWeights=None, pool_size=50, inertial=1.0,
              c1=1.0, c2=1.0, xi=1.0, generation=None):
        startTime = datetime.datetime.now()
        depot_num = idToDemandsLookup.shape[0]

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnCreate(_m):
            if _m == 0:
                _p = Position(np.zeros(depot_num), np.zeros(depot_num), truck_num)
                _v = Velocity(np.zeros(depot_num), np.zeros(depot_num))
                return _p, _v
            return create(truck_num, depot_num)

        def fnGetFitness(_p, _m):
            if _m == 0:
                return Fitness(float('inf'))
            return Fitness(get_fitness(_p, truck_num, truck_capacity,
                               idToLocLookup, idToDemandsLookup))

        optimalFitness = Fitness(optimalWeights)
        best, generation_mean_fitness, historical_best_fitness = particleSwarm.get_best(optimalFitness, fnGetFitness,
                                                                                        fnDisplay,
                                                                                        custom_generate=fnCreate,
                                                                                        poolSize=pool_size,
                                                                                        inertial=inertial,
                                                                                        c1=c1, c2=c2, xi=xi,
                                                                                        generation=generation)
        print("Optimal Solution: \n{}".format(best.Position))
        return best, generation_mean_fitness, historical_best_fitness


    def test_3_trucks_7_depos(self):
        idToLocLookup = np.array([
            [18, 54],
            [22, 60],
            [58, 69],
            [71, 71],
            [83, 46],
            [91, 38],
            [24, 42],
            [18, 40]
        ])
        idToDemandsLookup = np.array([
            0.89, 0.14, 0.28, 0.33, 0.21, 0.41, 0.57
        ])
        capacity=1.0
        truck_num=3
        opt_weights = 217.82

        particleSwarm.Benchmark.run(lambda pool_s=100, inertial=1.0, c1=1.0,
                                           c2=1.0, xi=1.0, generation=1000:
                                    self.solve(truck_num, capacity,
                                               idToLocLookup, idToDemandsLookup,
                                               optimalWeights=opt_weights, pool_size=pool_s, inertial=inertial,
                                               c1=c1, c2=c2, xi=xi, generation=generation),
                                    visualization=True)


if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    unittest.main()