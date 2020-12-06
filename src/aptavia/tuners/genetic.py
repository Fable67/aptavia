from simple_pid import PID
import numpy as np
import random


class GeneticTuner(object):
    """A PID Tuner using a genetic algorithm.

    :param population_size: How many random PID-Controllers are in the population, 
        defaults to 100
    :type population_size: int, optional
    :param mutation_probability: The probability of a gain of a PID-Controller in
        the population randomly changing, defaults to 0.2
    :type mutation_probability: float, optional
    :param fitness_function: A function that evaluates the fitness of one 
        PID-Controller, defaults to None
    :type fitness_function: function, optional
    """

    def __init__(self, population_size=100, mutation_probability=0.2, fitness_function=None):
        """Constructor method
        """
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.population = [PID(random.random(), random.random(), random.random())
                           for _ in range(population_size)]
        self.fitness_function = self.__fitness if fitness_function is None else fitness_function

    def set_population_size(self, population_size):
        """Setter for population_size

        :param population_size: How many random PID-Controllers are in the population
        :type population_size: int 
        """
        self.population_size = population_size

    def set_mutation_probability(self, mutation_probability):
        """Setter for mutation_probability

        :param mutation_probability: The probability of a gain of a PID-Controller in
            the population randomly changing
        :type mutation_probability: float 
        """
        self.mutation_probability = mutation_probability

    def set_population(self, population):
        """Setter for the population

        :param population: A list of PID_Controllers
        :type population: list of :class:`simple_pid.PID`
        """
        assert(len(population) == self.population_size)
        self.population = population

    def set_fitness_function(self, fitness_function):
        """Setter for fitness_function

        :param fitness_function: A function that evaluates the fitness of one 
            PID-Controller
        :type fitness_function: function
        """
        self.fitness_function = fitness_function

    def __fitness(self, agent):
        return random.random() * 100 - random.random() * 100

    def __mutate(self, agent):
        mutated_agent_tunings = list(
            map(
                lambda K: random.random() if random.random() < self.mutation_probability else K,
                agent.tunings
            )
        )
        agent.tunings = mutated_agent_tunings
        return agent

    def __crossover(self, agent1, agent2):
        child_tunings = list(map(lambda x: (x[0] + x[1]) * 0.5,
                                 zip(agent1.tunings, agent2.tunings)))
        return PID(*child_tunings)

    def __normalize(self, x):
        return x / x.sum()

    def step(self, num_generations=1):
        """Performs the genetic algorithm over a specified number of generations

        :param num_generations: The number of generations to tune, defaults to 1
        :type num_generations: int, optional
        """
        assert(num_generations > 0)
        assert(len(self.population) == self.population_size)
        for _ in range(num_generations):
            # calculate fitness of the generation
            population_fitness = [self.__fitness(agent) for agent in self.population]
            # select parents for breeding
            population_fitness_np = np.array(population_fitness)
            #   change scale of distribution to [0, min+max] if there is a fitness below 0
            if np.any(population_fitness_np < 0):
                population_fitness_np -= population_fitness_np.min()
                population_fitness_np += 1e-6
            #   make fitness distribution sum to 1
            population_fitness_np = self.__normalize(population_fitness_np)
            parents = np.random.choice(self.population, size=(
                self.population_size, 2), p=population_fitness_np)
            # Create new population by breeding parents
            self.population = list(map(lambda x: self.__crossover(x[0], x[1]), parents))
            # Mutate the new population
            self.population = list(map(self.__mutate, self.population))
