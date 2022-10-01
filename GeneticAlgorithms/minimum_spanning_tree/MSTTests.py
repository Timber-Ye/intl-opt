# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 16:16
# @Author  : Hanchiao
# @File    : MSTTests.py

import datetime
import random
import unittest
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import genetic


def display(candidate, startTime, c):
    timeDiff = datetime.datetime.now() - startTime
    length = len(candidate.Phenotype)
    if length < 10:
        print("{}\t{:.0f}\t{:<10}\t{}\n".format(
            '-'.join(map(str, candidate.Genes)),
            c / candidate.Fitness,
            candidate.Strategy.name,
            timeDiff))
    else:
        print("{}...{}\t{:.0f}\t{:<10}\t{}\n".format(
            '-'.join(map(str, candidate.Phenotype[:5])),
            '-'.join(map(str, candidate.Phenotype[-5:])),
            c / candidate.Fitness,
            candidate.Strategy.name,
            timeDiff))


def create(gene_set):
    return random.choices(gene_set, k=len(gene_set) - 2)


def repair(gene, pos, n):
    """
    修复非法（不满足度约束）的基因型
    :param gene:
    :param pos: 不满足度约束的节点的索引
    :param n:
    :return:
    """
    # print("gene:{}, pos:{}".format(gene, pos))
    p = random.choice(pos)
    alternatives = random.choice(list(range(n - 1)))
    gene[p] = alternatives \
        if alternatives != gene[p] \
        else n - 1
    return gene


def get_fitness(edge, adj_mat):
    """
    获得适应值，得到是生成树的加权和
    :param edge: 解码后得到的边集合
    :param adj_mat: 邻接矩阵
    :return:
    """
    fitness = 0
    for i in list(range(len(edge))):
        fitness += adj_mat[edge[i][0]][edge[i][1]]
    return fitness


def decode(genes, n):
    """
    解码过程
    :param genes: 基因
    :param n: 节点个数，即基因编码范围
    :return:
    """
    gene = genes[:]
    _gene = [x for x in list(range(n)) if not x in gene]
    # print("genes:{}, _genes:{}".format(gene, _gene))
    edge = []
    while len(gene) != 0:
        a = gene.pop(0)
        edge.append([a, _gene.pop(0)])
        if not a in gene:
            _gene.append(a)
    edge.append([_gene.pop(), _gene.pop()])

    return edge


def crossover(parentGenes, donorGenes):
    length = len(parentGenes)
    cutIndex = random.randrange(1, length)
    childGenes = parentGenes[:cutIndex] + donorGenes[cutIndex:]
    return childGenes


def mutate(genes, node_num, mutate_rate):
    """
    变异
    :param genes:
    :param node_num: 节点个数，即基因编码范围
    :param mutate_rate: 变异率
    :return:
    """
    childGenes = genes[:]

    '''Two different definitions of MUTATE_RATE'''

    # if random.uniform(0, 1) < mutate_rate:
    #     index = random.randrange(0, len(genes))
    #     newGene, alternative = random.sample(geneSet, 2)
    #     childGenes[index] = newGene \
    #         if newGene != childGenes[index] \
    #         else alternative
    # return childGenes

    for i in range(len(genes)):
        if random.uniform(0, 1) < mutate_rate:
            newGene, alternative = random.sample(list(range(node_num)), 2)
            childGenes[i] = newGene \
                if newGene != childGenes[i] \
                else alternative
    return childGenes


class MSTTests(unittest.TestCase):

    def solve(self, adj_mat, max_degree=None, optimalWeights=None,
              mutate_rate=0.01, crossover_rate=0.9,
              generation=None, poolSize=20, visualization=True):
        startTime = datetime.datetime.now()
        nodes_num = adj_mat.shape[0]
        gene_set = list(range(nodes_num))
        c = np.sum(adj_mat)/8

        def invalid_pos(gene):
            """
            判断基因型是否满足度约束
            :param gene:
            :return:
            """
            if max_degree is None:
                return None
            else:
                stat = dict(Counter(gene))
                dm = max(stat.values())
                if dm <= max_degree:
                    return None
                pos = []
                for k, v in stat.items():
                    if v == dm:
                        pos.extend([i for i, x in enumerate(gene) if x == k])
                        break
                return pos

        def fnRepair(gene, pos):
            return repair(gene, pos, nodes_num)

        def fnExpress(gene):
            return decode(gene, nodes_num)

        def fnDisplay(candidate):
            display(candidate, startTime, c)

        def fnCreate():
            gene = create(gene_set)
            pos = invalid_pos(gene)
            while pos is not None:  # repair
                gene = fnRepair(gene, pos)
                pos = invalid_pos(gene)
            return gene

        def fnGetFitness(genes):
            return c / get_fitness(genes, adj_mat)

        def fnMutate(genes):
            gene = mutate(genes, nodes_num, mutate_rate)
            pos = invalid_pos(gene)
            while pos is not None:  # repair
                gene = fnRepair(gene, pos)
                pos = invalid_pos(gene)
            return gene

        def fnCrossover(parentGenes, donorGenes):
            gene = crossover(parentGenes, donorGenes)
            pos = invalid_pos(gene)
            while pos is not None:  # repair
                gene = fnRepair(gene, pos)
                pos = invalid_pos(gene)
            return gene

        optimalFitness = c / optimalWeights
        best, generation_mean_fitness, historical_best_fitness = genetic.get_best(nodes_num, optimalFitness, fnExpress,
                                get_fitness=fnGetFitness, display=fnDisplay,
                                custom_crossover=fnCrossover, custom_mutate=fnMutate,
                                custom_create=fnCreate, poolSize=poolSize,
                                crossover_rate=crossover_rate, generation=generation)
        # visualization
        if visualization:
            x_axis = len(generation_mean_fitness)
            generation_mean_fitness = [c / x for x in generation_mean_fitness]
            historical_best_fitness = [c / x for x in historical_best_fitness]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # ax.set_aspect(1)
            ax.set(title='Find the minimum spanning tree',
                   xlabel='No. of generation', ylabel='fitness',
                   xlim=[0, x_axis])
            plt.grid(linestyle='--', linewidth=1, alpha=0.3)
            x_axis = list(range(x_axis))
            plt.plot(x_axis, generation_mean_fitness, color='b', label='mean fitness')
            plt.plot(x_axis, historical_best_fitness, color='g', label='best so far')
            plt.legend()
            plt.savefig('../../fig/[tmp]Find the minimum spanning tree.pdf')

        self.assertTrue(best is not None)
        self.assertTrue(best.Fitness >= optimalFitness)

    # def test_benchmark(self):
    #     genetic.Benchmark.run(lambda: self.solve(length=30, mutate_rate=0.01,
    #                                              crossover_rate=0.9, generation=None,
    #                                              poolSize=20))

    def test_9_nodes(self):
        adj_mat = np.array([[0, 224, 224, 361, 671, 300, 539, 800, 943],
                            [224, 0, 200, 200, 447, 283, 400, 728, 762],
                            [224, 200, 0, 400, 566, 447, 600, 922, 949],
                            [361, 200, 400, 0, 400, 200, 200, 539, 583],
                            [671, 447, 566, 400, 0, 600, 447, 781, 510],
                            [300, 283, 447, 200, 600, 0, 283, 500, 707],
                            [539, 400, 600, 200, 447, 283, 0, 361, 424],
                            [800, 728, 922, 539, 781, 500, 361, 0, 500],
                            [943, 762, 949, 583, 510, 707, 424, 500, 0]])
        max_degree = 3
        opt_weights = 2209
        genetic.Benchmark.run(lambda: self.solve(adj_mat, max_degree, opt_weights, mutate_rate=0.015,
                                                 crossover_rate=0.9, generation=2000,
                                                 poolSize=180, visualization=True))


if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    unittest.main()
