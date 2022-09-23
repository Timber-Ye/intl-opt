# specific to the one-max problem project
# file: oneMaxTests.py

import datetime
import random
import unittest

import genetic


def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    length = len(candidate.Genes)
    print("{}...{}\t{:3.2f}\t{}".format(
        ''.join(map(str, candidate.Genes if length < 25 else candidate.Genes[:10])),
        ''.join(map(str, candidate.Genes if length < 25 else candidate.Genes[-10:])),
        candidate.Fitness,
        timeDiff))


def get_fitness(genes):
    return genes.count(1)


class OneMaxTests(unittest.TestCase):
    geneSet = [0, 1]

    def test(self, length=100):
        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnGetFitness(genes):
            return get_fitness(genes)

        optimalFitness = length
        best = genetic.get_best(fnGetFitness, length, optimalFitness, self.geneSet, fnDisplay, iteration=1e5)
        self.assertEqual(best.Fitness, optimalFitness)

    def test_benchmark(self):
        genetic.Benchmark.run(lambda: self.test(4000))

