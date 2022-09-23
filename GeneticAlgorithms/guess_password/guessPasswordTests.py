# specific to the password guessing project
# File：guessPasswordTests.py

import datetime
import unittest

import genetic


def display(genes, target, startTime):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(genes, target)
    print("{}\t{}\t{}".format(genes, fitness, timeDiff))


def get_fitness(genes, target):
    return sum(1 for expected, actual in zip(target, genes) if expected == actual)


class GuessPasswordTests(unittest.TestCase):
    geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."

    def test_hello_world(self):
        target = 'Hello World!'
        self.guess_password(target)

    def test_For_I_am_fearfully_and_wonderfully_made(self):
        target = "For I am fearfully and wonderfully made."
        self.guess_password(target)

    def guess_password(self, target):
        startTime = datetime.datetime.now()

        def fnGetFitness(genes):
            return get_fitness(genes, target)

        def fnDisplay(genes):
            display(genes, target, startTime)

        optimalFitness = len(target)
        best = genetic.get_best(fnGetFitness, len(target), optimalFitness, self.geneSet, fnDisplay)

        self.assertEqual(best, target)


if __name__ == '__main__':
    unittest.main()
