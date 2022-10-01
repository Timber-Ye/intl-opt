# specific to the one-max problem project
# file: oneMaxTests.py

import datetime
import random
import unittest
import genetic


def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    length = len(candidate.Genes)
    if length < 25:
        print("{}\t{:^6}\t{:<10}\t{}\n".format(
            ' '.join(map(str, candidate.Genes)),
            candidate.Fitness,
            candidate.Strategy.name,
            timeDiff))
    else:
        print("{}...{}\t{:^6}\t{:<10}\t{}\n".format(
            ' '.join(map(str, candidate.Genes[:5])),
            ' '.join(map(str, candidate.Genes[-5:])),
            candidate.Fitness,
            candidate.Strategy.name,
            timeDiff))



def get_fitness(genes):
    return genes.count(1)


def crossover(parentGenes, donorGenes):
    length = len(parentGenes)
    cutIndex = random.randrange(1, length)
    childGenes = parentGenes[:cutIndex] + donorGenes[cutIndex:]
    return childGenes


def mutate(genes, geneSet, mutate_rate):
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
            newGene, alternative = random.sample(geneSet, 2)
            childGenes[i] = newGene \
                if newGene != childGenes[i] \
                else alternative
    return childGenes


class OneMaxTests(unittest.TestCase):
    geneSet = [0, 1]

    def solve(self, length=100, mutate_rate=0.01, crossover_rate=0.9,
              generation=None, poolSize=20):
        startTime = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, startTime)

        def fnGetFitness(genes):
            return get_fitness(genes)

        def fnMutate(genes):
            return mutate(genes, self.geneSet, mutate_rate)

        def fnCrossover(parentGenes, donorGenes):
            return crossover(parentGenes, donorGenes)

        optimalFitness = length
        best = genetic.get_best(fnGetFitness, length, optimalFitness, self.geneSet,
                                fnDisplay, crossover=fnCrossover, custom_mutate=fnMutate,
                                poolSize=poolSize, crossover_rate=crossover_rate,
                                generation=generation)
        self.assertEqual(best.Fitness, optimalFitness)

    def test_benchmark(self):
        genetic.Benchmark.run(lambda: self.solve(length=30, mutate_rate=0.01,
                                                 crossover_rate=0.9, generation=None,
                                                 poolSize=20))


if __name__ == '__main__':
    unittest.main()
