import copy
import math
import random
import time
from copy import deepcopy
import numpy as np
from Constants import Constants
import Reporter
from Solution import Solution

"""
TODO:
- Make code more efficient, can use time to benchmark efficiency.
- accessing local variables is typically faster than accessing atributes!
  --> so move field attributes to local variables in relevant functions when possible
- Try adaptive selection methods with exponential fall-off eg
- Make hyper parameters adaptive
- think about age methods
- elimination perhaps age based?, maybe add random stuff.
- As the algorithm proceed near the convergence we could add local search operators.
- Initially add more diversity promotion.
- Use different algorithms between early and late phases
- Transition between phases can be monitored by:
  diversity in fitness values (eg how much better is best vs the mean)
  Stagnation in best fitness
  Population diversity
- Edge Recombination crossover, Order crossover, Cycle crossover.
- Island model might be usefull for multithreading.

"""

class R0:
    """Setup of variables for the algorithm    """
    def __init__(self, filepath: str):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        with open(filepath) as file:
            self.distance_matrix: np.ndarray = np.loadtxt(file, delimiter=',')
        self.length: int = len(self.distance_matrix)
        self.lambdaa: int = 4  * self.length
        self.start_threshold: float = 0.4
        self.max_threshold: float = 0.9
        self.threshold: int = math.ceil(self.lambdaa * self.start_threshold)
        self.random_thresh : int = self.lambdaa - self.threshold
        self.mu: int = 2 * self.lambdaa
        self.concat : int = self.lambdaa + self.mu
        self.sigma : int = 50
        self.alpha : int = 0
        self.k: int = 3
        self.k_max: int = 5
        self.average_score: float = float('inf')
        # self.mean_improvement: float = 0
        self.beta: float = 0.5
        self.max_improvement: float = 1
        self.hyper_threshold: float = 0.01
        self.transition: bool = False
        # self.string: str = "exploration!"
        self.count: int = 0
        self.count_threshold: int = 5

    """ The main evolutionary algorithm loop."""
    def optimize(self):
        iteration_limit = 700
        time_limit = 60
        start_time = time.time()
        population = self.intialize_random()
        # for i in range(iteration_limit):
        while time.time() - start_time < time_limit:
        #     print("1",population)
        #     print(i)
            if not self.transition:
                self.calculate_fitness_sharing(population)
                parents = self.select_k_tournament_fitness(population)
                # print("2", parents)
                offspring = self.recombine_PMX(parents)
                # print("3", offspring)
                joined_population = np.concatenate((population, offspring))
                # print("joined:", joined_population)
                self.mutate_pop_and_insert(joined_population, self.beta)
                # print("mutated", joined_population)
                population = self.eliminate_diversity(joined_population)
                previous_average = self.average_score
                self.average_score = self.average(population)
                self.adapt_parameters(previous_average, self.average_score)
                # print(i, population[0].cost)
                # print("average cost:",self.average_score)
            else:
                parents = self.select_k_tournament_rep(population)
                # print("2", parents)
                offspring = self.recombine_PMX(parents)
                # print("3", offspring)
                joined_population = np.concatenate((population, offspring))
                # print("joined:", joined_population)
                self.mutate_pop_and_insert(joined_population, self.beta)
                # print("mutated", joined_population)
                population = self.eliminate(joined_population)
                # print(i, population[0].cost)
        return population

    def adapt_parameters(self, previous_average: np.floating, average: np.floating):
        # print(self.string)
        if previous_average != float('inf') and self.transition == False:
            improvement = previous_average - average
            if improvement > self.max_improvement:
                self.max_improvement = improvement
            ratio = improvement / self.max_improvement
            # print(improvement)
            # print(ratio)
            if ratio <= self.hyper_threshold:
                self.count += 1
            else:
                self.count = 0

            if self.count == self.count_threshold:
                # self.string = "Exploitation!!"
                self.transition = True
                self.k = self.k_max
                self.beta = 0.05
                self.start_threshold = self.max_threshold

    """ Initializes the population randomly."""
    def intialize_random(self) -> np.ndarray:
        initial_population = np.array([Solution(self.length) for _ in range(self.lambdaa)])
        for solution in initial_population:
            np.random.shuffle(solution.path[1:])
            self.calculate_cost(solution)
        return initial_population

    """ Performs k-tournament selection to select pairs of parents by repeated selection."""
    def select_k_tournament_rep(self, seed_population: np.ndarray) -> np.ndarray:
        selected = np.empty((self.mu, 2),dtype=Solution )
        for i in range(self.mu):
            parent_1 = self.tournament_selection(seed_population, 1)
            parent_2 = self.tournament_selection(seed_population, 1)
            parents = np.concatenate((parent_1, parent_2))
            selected[i] = parents
        return selected

    """ Performs k-tournament selection to select pairs of parents by repeated selection on fitness."""
    def select_k_tournament_fitness(self, seed_population: np.ndarray) -> np.ndarray:
        selected = np.empty((self.mu, 2),dtype=Solution )
        for i in range(self.mu):
            parent_1 = self.tournament_selection_fitness(seed_population, 1)
            parent_2 = self.tournament_selection_fitness(seed_population, 1)
            parents = np.concatenate((parent_1, parent_2))
            selected[i] = parents
        return selected

    def recombine_edge(self):
        return None


    """ Recombines using Partially Mapped Crossover"""
    def recombine_PMX(self, parent_population: np.ndarray) -> np.ndarray:
        offspring = np.empty(self.mu, dtype=Solution)
        offspring_pointer = 0
        for parents in parent_population:
            parent0_path = parents[0].path
            parent1_path = parents[1].path
            child = Solution(self.length)
            path = np.zeros(self.length, dtype=int)
            random_index1 = random.randint(1, self.length - 1)
            random_index2 = random.randint(1, self.length - 1)
            if random_index1 > random_index2:
                start = random_index2
                end = random_index1
            else:
                start = random_index1
                end = random_index2
            segment = parent0_path[start:end]
            path[start:end] = segment
            for i in range(start, end):
                number = parent1_path[i]
                if number not in segment:
                    temp = segment[i - start]
                    index = 0
                    while temp != 0:
                        index = np.where(parent1_path == temp)
                        temp = path[index]
                    path[index] = number
            for i in range(1, self.length):
                if path[i] == 0:
                    path[i] = parent1_path[i]
            child.path = path
            self.calculate_cost(child)
            offspring[offspring_pointer] = child
            offspring_pointer += 1
        return offspring


    """ Performs a mutation on the population with a certain chance a."""
    def mutate_pop_and_insert(self, population: np.ndarray,  chance: float) -> None:
        for solution in population:
            if random.random() < chance:
                self.pop_and_insert(solution)
                self.calculate_cost(solution)

    def mutate_swap(self, population: np.ndarray,  chance: float) -> None:
        for solution in population:
            if random.random() < chance:
                self.swap(solution)
                self.calculate_cost(solution)

    def mutate_scramble(self, population: np.ndarray, chance: float) -> None:
        for solution in population:
            random_index1 = random.randint(1, self.length - 1)
            random_index2 = random.randint(1, self.length - 1)
            if random_index1 > random_index2:
                start = random_index2
                end = random_index1
            else:
                start = random_index1
                end = random_index2
            subpath = np.zeros(end - start, dtype=int)
            index = np.arange(start, end)
            np.random.shuffle(index)
            pointer = 0
            for i in index:
                subpath[pointer] = population[i]
            solution.path = np.concatenate((solution.path[0:start], subpath, solution.path[end:]))

    """ Eliminates the unfit candidate solutions and cuts the total population back to lambda."""
    def eliminate(self, population: np.ndarray) -> np.ndarray:
        sorted_population = np.array(sorted(population, key=lambda solution: solution.cost))
        return sorted_population[:self.lambdaa]

    """ Elminates the unfit candidates but attempts to keep diversity."""
    def eliminate_diversity(self, population: np.ndarray) -> np.ndarray:
        sorted_population = np.array(sorted(population, key=lambda solution: solution.cost))
        randomAr = np.zeros(self.random_thresh, dtype=Solution)
        for i in range(0,self.random_thresh):
            randomAr[i] = sorted_population[random.randint(self.random_thresh, self.concat - 1)]

        return np.concatenate((sorted_population[:self.threshold], randomAr))


    # ================================
    # Helper functions
    # ================================
    """ Calculates and updates the cost of functions from scratch."""
    def calculate_cost(self, solution: Solution) -> None:
        path = solution.path
        cost = np.sum(self.distance_matrix[path[:-1], path[1:]])
        cost += self.distance_matrix[solution.path[-1], 0]
        solution.cost = cost
        # solution.cost_correct = True

    def calculate_fitness_sharing(self, population) -> None:
        for i in range(self.lambdaa):
            fitness = 0
            for j in range(self.lambdaa):
                if j != i:
                    fitness += 1 - self.hamming_distance(population[i], population[j])/self.sigma
            population[i].fitness = fitness * population[i].cost


    """ Performs k-tournament selection to select n parents from k samples."""
    def tournament_selection(self, seed_population: np.ndarray, n: int) -> np.ndarray:
        candidates = np.random.choice(self.lambdaa, self.k, replace=False)
        candidate_parents = np.array([seed_population[i] for i in candidates])
        sorted_parents = sorted(candidate_parents, key = lambda solution: solution.cost)
        return np.array(sorted_parents[:n])

    def tournament_selection_fitness(self, seed_population: np.ndarray, n: int) -> np.ndarray:
        candidates = np.random.choice(self.lambdaa, self.k, replace=False)
        candidate_parents = np.array([seed_population[i] for i in candidates])
        sorted_parents = sorted(candidate_parents, key=lambda solution: solution.fitness)
        return np.array(sorted_parents[:n])

    """ Removes an element from the path at random at places it somewhere again at random"""
    def pop_and_insert(self, solution: Solution) -> None:
        path = solution.path
        random_index1 = random.randint(1, self.length - 1)
        random_index2 = random.randint(1, self.length - 1)
        if random_index1 < random_index2:
            solution.path = np.concatenate((path[0:random_index1],
                                            path[random_index1 + 1: random_index2],
                                            np.array([path[random_index1]]),
                                            path[random_index2:]))
        else:
            solution.path = np.concatenate((path[0:random_index2],
                                            np.array([path[random_index1]]),
                                            path[random_index2: random_index1],
                                            path[random_index1+ 1:]))

    """ Randomly swaps two elements of the path."""
    def swap(self, solution: Solution) -> None:
        random_index1 = random.randint(1, self.length - 1)
        random_index2 = random.randint(1, self.length - 1)
        solution.path[random_index1], solution.path[random_index2] = solution.path[random_index2], solution.path[random_index1]

    """ Takes a mutation method, if path was originaly infinity it will aply it once. if the path was no infinity
        it will reapply if the mutation makes it infinity."""
    def reaply_infinity(self, function, solution: Solution) -> None:
        if solution.cost == float('inf'):
            function(solution)
            self.calculate_cost(solution)
        else:
            original_path = copy.deepcopy(solution.path)
            function(solution)
            self.calculate_cost(solution)
            while solution.cost == float('inf'):
                solution.path = copy.deepcopy(original_path)
                function(solution)
                self.calculate_cost(solution)

    def average(self,population: np.ndarray) -> np.floating:
        costs = np.array([solution.cost for solution in population if solution.cost != float('inf')])
        return np.mean(costs)

    def hamming_distance(self, solution1: Solution, solution2: Solution) -> int:
        distance: int = 0
        for i in range(self.length):
            if solution1.path[i] != solution2.path[i]:
                distance += 1
        return distance


def tester(size: int):
    scores = np.zeros(size)
    for i in range(size):
        algorithm = R0(Constants.FILE50)
        population = algorithm.optimize()
        scores[i] = population[0].cost
        print("Best solution iteration: ",i + 1,population[0])
    average_score = np.mean(scores)
    print("Average score: ", average_score)


def __main__():
    tester(5)

__main__()
















