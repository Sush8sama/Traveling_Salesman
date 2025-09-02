import copy
import math
import random
import time
from multiprocessing import Pool
import numpy as np

import Reporter
from Constants import Constants



""" Representation of a solution."""
class Solution:
    def __init__(self, length: int):
        self.path: np.ndarray = np.arange(length, dtype=int)
        self.cost: int = 0
        self.fitness: int = 0



    def __repr__(self):
        return f"cost: {self.cost}, path: {self.path}"

    def __str__(self):
        return f"cost: {self.cost}, path: {self.path}"

""" Representation of the group of solution with its hyper parameters."""
class Island:
    def __init__(self,
                 index: int,
                 distance_matrix: np.ndarray,
                 length: int,
                 lambdaa: int,
                 migrant_amount: int,
                 mu: int,
                 beta: float,
                 k: int,
                 step_k: int,
                 start_k : int,
                 sigma: int,
                 start_threshold: float,
                 max_threshold: float,
                 threshold_ratio: float,
                 threshold: int,
                 random_threshold: int,
                 concat: int,
                 time_limit: int,
                 start_time: float,
                 greedy: bool,
                 greedy_amount: int,
                 greedy_k : int,
                 last_k: int,
                 search_population: int,
                 ):
        self.index: int = index
        self.distance_matrix: np.ndarray = distance_matrix
        self.length: int = length
        self.lambdaa: int = lambdaa
        self.migrant_amount: int = migrant_amount
        self.mu : int = mu
        self.beta : float = beta
        self.k : int = k
        self.step_k : int = step_k
        self.start_k : int = start_k
        self.sigma : int = sigma
        self.start_threshold : float = start_threshold
        self.max_threshold : float = max_threshold
        self.threshold : int = threshold
        self.random_thresh : int = random_threshold
        self.threshold_ratio : float = threshold_ratio
        self.concat : int = concat
        self.population: np.ndarray = np.empty(1)
        self.time_limit: int = time_limit
        self.start_time: float = start_time
        self.greedy: bool = greedy
        self.greedy_amount: int = greedy_amount
        self.greedy_k : int = greedy_k
        self.last_k : int = last_k
        self.search_population: int = search_population


    def initialize_random(self) -> None:
        island = [Solution(self.length) for _ in range(self.start)]
        for solution in island:
            np.random.shuffle(solution.path[1:])
            self.calculate_cost(solution)
        self.population = island


    def initialize_greedy(self) -> None:
        island = [Solution(self.length) for _ in range(self.lambdaa)]
        for i in range(self.greedy_amount):
            self.greedy_path(island[i])
            self.calculate_cost(island[i])
        for j in range(self.greedy_amount, self.lambdaa):
            np.random.shuffle(island[j].path[1:])
            self.calculate_cost(island[j])
        self.population = island

    def greedy_path(self, solution: Solution) -> None:
        left_over = solution.path[1:].tolist()
        for i in range(self.length - 1):

            num_candidates = min(self.greedy_k, len(left_over))
            candidates = random.sample(left_over, num_candidates)
            best_candidate = candidates[0]
            best_cost = self.distance_matrix[solution.path[i], best_candidate]
            for candidate in candidates[1:]:
                current_cost = self.distance_matrix[solution.path[i], candidate]
                if current_cost < best_cost:
                    best_candidate = candidate
                    best_cost = current_cost

            solution.path[i + 1] = best_candidate
            left_over.remove(best_candidate)


    """ Performs k-tournament selection to select pairs of parents by repeated selection."""
    def select_k_tournament_rep(self, seed_population: np.ndarray) -> np.ndarray:
        selected = np.empty((self.mu, 2), dtype=Solution)
        for i in range(self.mu):
            parent_1 = self.tournament_selection(seed_population, 1)
            parent_2 = self.tournament_selection(seed_population, 1)
            parents = np.concatenate((parent_1, parent_2))
            selected[i] = parents
        return selected

    """ Performs k-tournament selection to select pairs of parents by repeated selection on fitness."""
    def select_k_tournament_fitness(self, seed_population: np.ndarray) -> np.ndarray:
        selected = np.empty((self.mu, 2), dtype=Solution)
        for i in range(self.mu):
            parent_1 = self.tournament_selection_fitness(seed_population, 1)
            parent_2 = self.tournament_selection_fitness(seed_population, 1)
            selected[i,0] = parent_1[0]
            selected[i,1] = parent_2[0]
        return selected

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

    """ Performs insert mutation on the population with a certain chance."""
    def mutate_pop_and_insert(self, population: np.ndarray, chance: float) -> None:
        for solution in population:
            if random.random() < chance:
                self.pop_and_insert(solution)


    """ Performs swap mutation on the population with a certain chance."""
    def mutate_swap(self, population: np.ndarray,  chance: float) -> None:
        for solution in population:
            if random.random() < chance:
                self.swap(solution)


    """ Eliminates the unfit candidate solutions and cuts the total population back to lambda using top_k."""
    def eliminate(self, population: np.ndarray) -> np.ndarray:
        sorted_population = np.array(sorted(population, key=lambda solution: solution.cost))
        return sorted_population[:self.lambdaa]

    """ Elminates the unfit candidates but attempts to keep diversity by using top_k but adds some randomness."""
    def eliminate_diversity(self, population: np.ndarray) -> np.ndarray:
        sorted_population = np.array(sorted(population, key=lambda solution: solution.cost))
        randomAr = np.zeros(self.random_thresh, dtype=Solution)
        randomArIndex = np.random.choice(np.arange(self.threshold, self.concat),size=self.random_thresh, replace=False)
        for i in range(0, self.random_thresh):
            randomAr[i] = sorted_population[randomArIndex[i]]
        return np.concatenate((sorted_population[:self.threshold], randomAr))

    # ================================
    # Helper functions
    # ================================
    """ Calculates and updates the cost of functions from scratch."""
    def calculate_cost(self, solution: Solution) -> None:
        path = solution.path
        cost = np.sum(self.distance_matrix[path[:-1], path[1:]])
        cost += self.distance_matrix[path[-1], 0]
        solution.cost = cost
        # solution.cost_correct =

    """ Same as above but on a path and returns the cost."""
    def calculate_cost_path(self, path: np.ndarray) -> float:
        cost = np.sum(self.distance_matrix[path[:-1], path[1:]])
        cost += self.distance_matrix[path[-1], 0]
        return cost

    """ Calculates the fitness using the fitness sharing formula as seen in the lectures."""
    def calculate_fitness_sharing(self, population : np.ndarray[Solution], sigma: int, alpha: int) -> None:
        for i in range(self.lambdaa):
            fitness = 0
            for j in range(self.lambdaa):
                if j != i:
                    fitness += 1 - (self.hamming_distance(population[i], population[j])/sigma)**alpha
            if population[i].cost != float('inf'):
                population[i].fitness = population[i].cost * fitness
            else:
                population[i].fitness = fitness


    """ Performs k-tournament selection to select n parents from k samples."""
    def tournament_selection(self, seed_population: np.ndarray, n: int) -> np.ndarray:
        candidates = np.random.choice(self.lambdaa, self.k, replace=False)
        candidate_parents = np.array([seed_population[i] for i in candidates])
        sorted_parents = sorted(candidate_parents, key=lambda solution: solution.cost)
        return np.array(sorted_parents[:n])

    """ Performs k-tournament selection to select n parents from k samples, using the fitness instead of the cost."""
    def tournament_selection_fitness(self, seed_population: np.ndarray, n: int) -> np.ndarray:
        candidates = np.random.choice(self.lambdaa, self.k, replace=False)
        candidate_parents = np.array([seed_population[i] for i in candidates])
        if all(solution.cost == float('inf') for solution in candidate_parents):
            sorted_parents = sorted(candidate_parents, key=lambda solution: solution.fitness)
        else:
            candidate_parents = np.array([candidate_parents[i] for i in range(len(candidate_parents))
                                         if candidate_parents[i].cost != float('inf')])
            sorted_parents = sorted(candidate_parents, key=lambda solution: solution.fitness)

        return np.array(sorted_parents[:n])

    """ Removes an element from the path at random at places it in front of another random index."""
    def pop_and_insert(self, solution: Solution) -> None:
        path = solution.path.copy()

        random_index1 = random.randint(1, self.length - 1)
        random_index2 = random.randint(1, self.length)
        while random_index1 == random_index2 - 1 or random_index1 == random_index2 : #Guarantee mutation.
            random_index1 = random.randint(1, self.length - 1)
            random_index2 = random.randint(1, self.length)

        if random_index1 < random_index2: #Moves index1 backwards
            if random_index2 == self.length: #Moves element to last place
                solution.path = np.concatenate((path[0:random_index1],
                                                path[random_index1 + 1: random_index2],
                                                np.array([path[random_index1]]),
                                                ))
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index1 - 1], path[random_index1 + 1]],
                                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]],
                                                     self.distance_matrix[path[random_index1], path[0]]]):
                    solution.cost = float('inf')

                elif solution.cost == float('inf'):
                    self.calculate_cost(solution)

                else:
                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index2 - 1], path[0]] -
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1]] -
                                     self.distance_matrix[path[random_index1],path[random_index1 + 1]] +
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1 + 1]] +
                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index1], path[0]]
                                     )

            else: #Moves element before in place index2
                solution.path = np.concatenate((path[0:random_index1],
                                            path[random_index1 + 1: random_index2],
                                            np.array([path[random_index1]]),
                                            path[random_index2:]))
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index1 - 1], path[random_index1 + 1]],
                                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]],
                                                     self.distance_matrix[path[random_index1], path[random_index2]]]):
                    solution.cost = float('inf')
                elif solution.cost == float('inf'):
                    self.calculate_cost(solution)
                else:
                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1]] -
                                     self.distance_matrix[path[random_index1], path[random_index1 + 1]] +
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1 + 1]] +
                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index1], path[random_index2]] -
                                     self.distance_matrix[path[random_index2 - 1],path[random_index2]]
                                     )

        else: #Moves index1 forwards
            if random_index1 == self.length - 1: #Moves the last element forward
                solution.path = np.concatenate((path[0:random_index2],
                                                np.array([path[random_index1]]),
                                                path[random_index2: random_index1]))
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index2 - 1], path[random_index1]],
                                                     self.distance_matrix[path[random_index1], path[random_index2]],
                                                     self.distance_matrix[path[random_index1 - 1], path[0]]]):
                    solution.cost = float("inf")
                elif solution.cost == float('inf'):
                    self.calculate_cost(solution)
                else:
                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index1], path[0]] -
                                     self.distance_matrix[path[random_index2 - 1], path[random_index2]] +
                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index1], path[random_index2]] -
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index1 - 1], path[0]]
                                     )
            else:
                solution.path = np.concatenate((path[0:random_index2],
                                                np.array([path[random_index1]]),
                                                path[random_index2: random_index1],
                                                path[random_index1 + 1:]))
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index2 - 1], path[random_index1]],
                                                     self.distance_matrix[path[random_index1], path[random_index2]],
                                                     self.distance_matrix[path[random_index1 - 1], path[random_index1 + 1]]]):
                    solution.cost = float("inf")
                elif solution.cost == float('inf'):
                    self.calculate_cost(solution)
                else:
                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index2 - 1], path[random_index2]] +
                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index1], path[random_index2]] -
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1]] -
                                     self.distance_matrix[path[random_index1], path[random_index1 + 1]] +
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1 + 1]]
                                     )

    """ Randomly swaps two elements of the path."""
    def swap(self, solution: Solution) -> None:
        path = solution.path.copy()

        random_index1, random_index2 = random.sample(range(1, self.length), 2)
        solution.path[random_index1], solution.path[random_index2] = solution.path[random_index2], solution.path[random_index1]
        if solution.cost == float('inf'):
            self.calculate_cost(solution)
        elif any(np.isinf(value) for value in [self.distance_matrix[path[random_index1 - 1], path[random_index2]],
                                             self.distance_matrix[path[random_index2 - 1], path[random_index1]]]):
            solution.cost = float('inf')
        elif random_index1 == self.length - 1: #Random1 is at the end
            if random_index1 == random_index2 + 1:
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index1], path[random_index2]],
                                                     self.distance_matrix[path[random_index2], path[0]]]):
                    solution.cost = float('inf')
                else:
                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index2 - 1], path[random_index2]] -
                                     self.distance_matrix[path[random_index2], path[random_index1]] -
                                     self.distance_matrix[path[random_index1], path[0]] +
                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index1], path[random_index2]] +
                                     self.distance_matrix[path[random_index2], path[0]]
                                     )
            else:
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index1], path[random_index2 + 1]],
                                                     self.distance_matrix[path[random_index2], path[0]]]):
                    solution.cost = float('inf')
                else:
                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index1], path[0]] -
                                     self.distance_matrix[path[random_index2 - 1], path[random_index2]] -
                                     self.distance_matrix[path[random_index2], path[random_index2 + 1]] +
                                     self.distance_matrix[path[random_index1 - 1], path[random_index2]] -
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                     self.distance_matrix[path[random_index1], path[random_index2 + 1]] +
                                     self.distance_matrix[path[random_index2], path[0]]
                                     )

        elif random_index2 == self.length - 1:

            if random_index1 == random_index2 - 1:
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index2], path[random_index1]],
                                                     self.distance_matrix[path[random_index1], path[0]]]):
                    solution.cost = float('inf')
                else:
                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1]] -
                                     self.distance_matrix[path[random_index1], path[random_index2]] -
                                     self.distance_matrix[path[random_index2], path[0]] +
                                     self.distance_matrix[path[random_index1 - 1], path[random_index2]] +
                                     self.distance_matrix[path[random_index2], path[random_index1]] +
                                     self.distance_matrix[path[random_index1], path[0]]
                                     )

            else:
                if any(np.isinf(value) for value in [self.distance_matrix[path[random_index2], path[random_index1 + 1]],
                                                     self.distance_matrix[path[random_index1], path[0]]]):
                    solution.cost = float('inf')
                else:

                    solution.cost = (solution.cost -
                                     self.distance_matrix[path[random_index2], path[0]] -
                                     self.distance_matrix[path[random_index1 - 1], path[random_index1]] -
                                     self.distance_matrix[path[random_index1], path[random_index1 + 1]] +
                                     self.distance_matrix[path[random_index2 - 1], path[random_index1]] -
                                     self.distance_matrix[path[random_index2 - 1], path[random_index2]] +
                                     self.distance_matrix[path[random_index1 - 1], path[random_index2]] +
                                     self.distance_matrix[path[random_index2], path[random_index1 + 1]] +
                                     self.distance_matrix[path[random_index1], path[0]]
                                     )
        elif random_index1 == random_index2 - 1:
            if any(np.isinf(value) for value in [self.distance_matrix[path[random_index2], path[random_index1]],
                                                 self.distance_matrix[path[random_index1], path[random_index2 + 1]]]):
                solution.cost = float('inf')
            else:
                solution.cost = (solution.cost -
                                 self.distance_matrix[path[random_index1 - 1], path[random_index1]] -
                                 self.distance_matrix[path[random_index1], path[random_index2]] -
                                 self.distance_matrix[path[random_index2], path[random_index2 + 1]] +
                                 self.distance_matrix[path[random_index1 - 1], path[random_index2]] +
                                 self.distance_matrix[path[random_index2], path[random_index1]] +
                                 self.distance_matrix[path[random_index1], path[random_index2 + 1]]
                                 )
        elif random_index1 == random_index2 + 1:
            if any(np.isinf(value) for value in [self.distance_matrix[path[random_index1], path[random_index2]],
                                                 self.distance_matrix[path[random_index2], path[random_index1 + 1]]]):
                solution.cost = float('inf')
            else:
                solution.cost = (solution.cost -
                                 self.distance_matrix[path[random_index2 - 1], path[random_index2]] -
                                 self.distance_matrix[path[random_index2], path[random_index1]] -
                                 self.distance_matrix[path[random_index1], path[random_index1 + 1]] +
                                 self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                 self.distance_matrix[path[random_index1], path[random_index2]] +
                                 self.distance_matrix[path[random_index2], path[random_index1 + 1]]
                                 )

        else:
            if any(np.isinf(value) for value in [self.distance_matrix[path[random_index2], path[random_index1 + 1]],
                                                 self.distance_matrix[path[random_index1], path[random_index2 + 1]]]):
                solution.cost = float('inf')
            else:
                solution.cost = (solution.cost -
                                 self.distance_matrix[path[random_index1 - 1], path[random_index1]] -
                                 self.distance_matrix[path[random_index1], path[random_index1 + 1]] +
                                 self.distance_matrix[path[random_index2 - 1], path[random_index1]] +
                                 self.distance_matrix[path[random_index1], path[random_index2 + 1]] -
                                 self.distance_matrix[path[random_index2 - 1], path[random_index2]] -
                                 self.distance_matrix[path[random_index2], path[random_index2 + 1]] +
                                 self.distance_matrix[path[random_index1 - 1], path[random_index2]] +
                                 self.distance_matrix[path[random_index2], path[random_index1 + 1]]
                                 )

    
    """ Performs multi swap mutation on the population with a certain chance."""
    def multi_swap_mutation(self, population: np.ndarray, chance: float, amount: int) -> None:
        for solution in population:
            if random.random() < chance:
                for _ in range(amount):
                    self.swap(solution)




    """ Calculates the hamming distances between two solutions."""
    def hamming_distance(self, solution1: Solution, solution2: Solution) -> int:
        distance: int = 0
        for i in range(self.length):
            if solution1.path[i] != solution2.path[i]:
                distance += 1
        return distance

    """ Local search operator that goes through all neighbors with 2 edges difference and chooses the best one."""
    def two_opt(self, population: np.ndarray):
        for solution in population:
            if random.random() < 1:
                best_solution_path = solution.path
                best_cost = solution.cost
                path = solution.path.copy()
                cost = solution.cost

                if cost != float('inf'):
                    for i in range(1, self.length - 1):
                        for j in range(i + 1, self.length):
                            new_path = np.concatenate((path[:i], np.flip(path[i: j]), path[j:]))
                            # new_cost = self.calculate_cost_path(new_path)
                            # if new_cost < best_cost:
                            #     best_solution_path = new_path
                            if not any(np.isinf(value) for value in [self.distance_matrix[path[i - 1], path[j - 1]] ,
                                                                 self.distance_matrix[path[i], path[j]]]):
                                new_cost = self.calculate_cost_path(new_path)
                                if new_cost < best_cost:
                                    best_solution_path = new_path

                solution.path = best_solution_path
                solution.cost = best_cost


# ================================
# Optimization architectures
# ================================

""" Initial loop for diversifying the starting population, fitness sharing function has heavier weight."""
def diversify(args):
    # print("here!")
    island, diversify_duration, scramble = args
    if island.greedy:
        island.initialize_greedy()
    else:
        island.initialize_random()
    island_population = island.population
    diverse_sigma = island.sigma
    for _ in range(diversify_duration):
        island.calculate_fitness_sharing(island_population, diverse_sigma, 2)
        parents = island.select_k_tournament_fitness(island_population)
        offspring = island.recombine_PMX(parents)
        joined_population = np.concatenate((island_population, offspring))
        island.multi_swap_mutation(joined_population, 0.9, scramble)
        island_population = island.eliminate_diversity(joined_population)
    island.population = island_population
    non_infinite_solutions = [solution.cost for solution in island.population if solution.cost != float('inf')]
    if len(non_infinite_solutions) == 0:
        average = float('inf')
    else:
        average = np.mean(non_infinite_solutions)
        # print(average, len(non_infinite_solutions), "here!!!")


    # print("Done diversifying!")
    return [island, average]

""" the localized early optimization loop for an island within isolation, using fitness sharing."""
def early_search(args):
    island, migration_interval_o, multiswap, _, search_k, _ = args
    sigma = island.sigma
    island_population = island.population
    island.k = search_k
    island.lambdaa =  island.search_population
    island.mu = island.search_population * 2
    island.threshold = math.ceil(island.lambdaa * island.start_threshold)
    island.random_thresh = island.lambdaa - island.threshold
    island.concat = island.lambdaa + island.mu
    for _ in range(migration_interval_o):
        island.calculate_fitness_sharing(island_population, sigma, 1)
        parents = island.select_k_tournament_fitness(island_population)
        offspring = island.recombine_PMX(parents)
        joined_population = np.concatenate((island_population, offspring))
        island.multi_swap_mutation(joined_population, island.beta, multiswap)
        island_population = island.eliminate_diversity(joined_population)
    island.population = island_population
    non_infinite_solutions = [solution.cost for solution in island.population if solution.cost != float('inf')]
    if len(non_infinite_solutions) == 0:
        average = float('inf')
    else:
        average = np.mean(non_infinite_solutions)
    return [island, average]

""" the localized early optimization loop for an island within isolation, optimized for big tours."""
def early_search_big(args):
    island, migration_interval_o, multiswap, _,  search_k, _ = args
    island_population = island.population
    island.k = search_k
    island.lambdaa =  island.search_population
    island.mu = island.search_population * 2
    island.threshold = math.ceil(island.lambdaa * island.start_threshold)
    island.random_thresh = island.lambdaa - island.threshold
    island.concat = island.lambdaa + island.mu
    # print("Early search started!", len(island_population))
    start_time = island.start_time
    time_limit = island.time_limit
    for _ in range(migration_interval_o):
        parents = island.select_k_tournament_rep(island_population)
        offspring = island.recombine_PMX(parents)
        joined_population = np.concatenate((island_population, offspring))
        island.multi_swap_mutation(joined_population, island.beta, multiswap)
        island_population = island.eliminate_diversity(joined_population)
        if time.time() - start_time > time_limit - 5:
            break
    island.population = island_population
    non_infinite_solutions = [solution.cost for solution in island.population if solution.cost != float('inf')]
    if len(non_infinite_solutions) == 0:
        average = float('inf')
    else:
        average = np.mean(non_infinite_solutions)
    return [island, average]


""" Final loop phase for finding the best solution."""
def local_search(args):
    # print("Local search started!")
    island, migration_interval_o, multiswap, local_search_population , _, two_opt= args
    island.lambdaa = local_search_population
    island.mu = island.lambdaa * 2
    island.threshold = math.ceil(island.lambdaa * island.start_threshold)
    island.random_thresh = island.lambdaa - island.threshold
    island.concat = island.lambdaa + island.mu

    island_population = island.eliminate(island.population)
    start_time = island.start_time
    time_limit = island.time_limit
    for _ in range(migration_interval_o):

        parents = island.select_k_tournament_rep(island_population)
        offspring = island.recombine_PMX(parents)
        joined_population = np.concatenate((island_population, offspring))
        island.mutate_pop_and_insert(joined_population, island.beta)
        island_population = island.eliminate(joined_population)
        if time.time() - start_time > time_limit - 5:
            break
        island.two_opt(island_population[:two_opt])
        if time.time() - start_time > time_limit - 5:
            break


    island.population = island_population
    average = np.mean([solution.cost for solution in island.population if solution.cost != float('inf')])
    return [island, average]



""" migrates the best paths to the next island in the ring, these paths will replace the worst paths."""
def migrate_elite(islands, island_amount: int) -> None:
    island_amount_i = island_amount
    migrant_amount = islands[0].migrant_amount
    island_size = islands[0].lambdaa
    islands[0].population[-migrant_amount:] = copy.deepcopy(islands[-1].population[:migrant_amount])
    for i in range(1,island_amount_i):
        islands[i].population[-migrant_amount:] = copy.deepcopy(islands[i-1].population[:migrant_amount])

def adapt_parameters(islands: np.ndarray[Island],
                     island_amount: int,
                     generation: int,
                     ratio: float,
                     a: int,
                     b: float,
                     grace_period: int,
                     stage: int) -> None:


    if stage == 0:
        beta = max(0.05, islands[0].beta * np.exp(-0.01 * generation))
        old_k = islands[0].k
        step_k = islands[0].step_k
        start_k = islands[0].start_k
        start_threshold = islands[0].start_threshold
        max_threshold = islands[0].max_threshold
        threshold_ratio = islands[0].threshold_ratio

        for i in range(island_amount):
            islands[i].beta = beta
        if generation > grace_period:
            new_threshold_ratio = max(threshold_ratio, start_threshold + max_threshold * 1 / (1 + np.exp(a * (ratio - b))))
            new_k = max(old_k, math.floor(start_k + step_k * 1 / (1 + np.exp(a * (ratio - b)))))
            new_threshold = math.ceil(islands[0].lambdaa * new_threshold_ratio)
            new_random_thresh = islands[0].lambdaa - new_threshold
            for i in range(island_amount):
                islands[i].k = new_k
                islands[i].threshold_ratio = new_threshold_ratio
                islands[i].threshold = new_threshold
                islands[i].random_threshold = new_random_thresh
    else:
        for i in range(island_amount):
            islands[i].k = islands[i].last_k


class r0885724:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    """ Main loop of the island model algorithm."""
    def optimize(self, filename):
        start_time = time.time()
        ##############
        # Parameters #
        ##############
        with open(filename) as file:
            distance_matrix: np.ndarray = np.loadtxt(file, delimiter=',')
        
        

        tour = len(distance_matrix)
        scramble = math.ceil(tour * 0.005) #multiswap amount during the diversification phase
        search_population = 100 #Population size for the search phase
        lambdaa = 300 # Starting population size
        migrant_ratio = 0.3
        migrant_amount = math.floor(lambdaa * migrant_ratio) #Amount that migrates to the next island
        mu = 2 * lambdaa  # Offspring size
        beta = 0.01  # beta minimum is 0.05, chance of mutation
        start_k = 8 # Initial k used for the diversification phase
        step_k = 2  # maximum k - starting k + 1
        search_k = 2 # Value after the diversification phase
        k = start_k  # starting tournament size
        last_k = 2 # value used for the local search phase
        sigma = tour  # fitness sharing distance metric
        # alpha
        start_threshold = 0.4  # amount that eliminate will choose using elite.
        max_threshold = 0.3  # acutall max will be max + start
        threshold_ratio = start_threshold
        threshold: int = math.ceil(lambdaa * start_threshold)
        random_thresh: int = lambdaa - threshold  # Amount that will be randomly choosen by eliminate
        concat: int = lambdaa + mu

        migration_interval = 100 # Interval between migrations
        diversify_duration = 1  #amount of iterations for the diversification phase
        island_amount = 2  # parallel concurrency, set to 2 due to the core limit of 2.
        max_iterations = 99999
        count = 0  # Used for stage switch
        max_count = 1  # After reaching stage_ratio this amount of time it will switch to local search
        stage_ratio = 0.06  # Threshold deciding when to local search
        # Paramter k adaptive
        a = 50  # steepness of sigmoid function for k
        b = 0.15  # transition point of sigmoid function for k
        grace_period = diversify_duration # Iterations where k is imuune to change during the second stage
        time_limit = 300
        greedy = True # Whether to use greedy initialization or random initialization
        greedy_amount = lambdaa//2  # Amount of greedy solutions in the population
        greedy_k = int(0.6 * tour) # math.floor(0.1 * length)

        local_search_population = 20  # Amount of solutions used in the local search phase

        option1 = [early_search_big, local_search]
        option2 = [early_search, local_search]
        stage = 0
        convergence_counter = 0
        convergence_limit = 2
        multiswap =1 + math.ceil(tour * 0.03) #mutliswap amount during the search phase for large tours

        islands = [
            Island(i, distance_matrix, tour, lambdaa, migrant_amount,
                   mu, beta, k, step_k, start_k, sigma, start_threshold, max_threshold,
                   threshold_ratio, threshold, random_thresh, concat, time_limit, start_time,
                   greedy, greedy_amount, greedy_k, last_k, search_population)
            for i in range(island_amount)
        ]
        
        #Initialization 
        if tour < 200:
            stages = option2
            multiswap = 1
            two_opt = 5
            with Pool(island_amount) as pool:  
                results = pool.map(
                diversify,
                [(islands[i], diversify_duration, scramble) for i in range(island_amount)]
                )
        else: 
            stages = option1
            two_opt = 1
            with Pool(island_amount) as pool:
                results = pool.map(
                diversify,
                [(islands[i], diversify_duration, scramble) for i in range(island_amount)]
                )
        

        # Collecting results from the initialization
        islands = [results[i][0] for i in range(island_amount)]
        non_infinite_averages = [results[i][1] for i in range(island_amount) if results[i][1] != float('inf')]

        if len(non_infinite_averages) == 0:
            global_average = float('inf')
        else:
            global_average = np.mean(non_infinite_averages)
        max_improvement = -float('inf')

        best_solution = islands[0].population[0]
        best_cost = best_solution.cost

        for i in range(1, island_amount):
                if islands[i].population[0].cost < best_cost:
                    best_solution = islands[i].population[0]
                    best_cost = best_solution.cost
        self.reporter.report(global_average, best_cost, best_solution.path)

        #Setting up the Optimization loops
        for iteration in range(0, max_iterations, migration_interval):  

            if stage == 0:
                if count == max_count:
                    stage += 1

            #Optimization loop
            with Pool(island_amount) as pool:
                results = pool.map(
                    stages[stage],
                    [(islands[i], migration_interval, multiswap, local_search_population, search_k, two_opt) for i in range(island_amount)]
                )
            islands = np.array([results[i][0] for i in range(island_amount)])
            best_solution = islands[0].population[0]
            best_cost = best_solution.cost

            
            #Data observation and performance evaluation
            for i in range(1, island_amount):
                if islands[i].population[0].cost < best_cost:
                    best_solution = islands[i].population[0]
                    best_cost = best_solution.cost

            
            non_infinite_averages = [results[i][1] for i in range(island_amount) if results[i][1] != float('inf')]
            if len(non_infinite_averages) == 0:
                new_global_average = float('inf')
            else:
                new_global_average = np.mean(non_infinite_averages)
            if new_global_average == float('inf') or global_average == float('inf'):
                global_average = new_global_average
                ratio = 1
                global_improvement = 0
            else:
                global_improvement = global_average - new_global_average
                global_average = new_global_average
                if global_improvement > max_improvement:
                    max_improvement = global_improvement
                ratio = max(0.001, global_improvement / max_improvement)

            #Parameter adapt    
            if stage == 0:
                adapt_parameters(islands, island_amount, iteration, ratio, a, b, grace_period, stage)
                if ratio < stage_ratio:
                    count += 1
                else:
                    count = 0

            #Migration and convergence check
            migrate_elite(islands, island_amount)
            time_left = self.reporter.report(global_average, best_cost, best_solution.path)
            if time_left < 0:
                break
            if ratio == 0.001:
                convergence_counter += 1
            else:
                convergence_counter = 0
            if convergence_counter >= convergence_limit:
                # print("Convergence reached, stopping optimization.")
                break
            # print("Best solution: ", islands[0].population[0].cost, islands[0].beta, islands[0].k, islands[0].threshold, ratio, stage)
        continent = np.concatenate([islands[i].population for i in range(island_amount)])
        sorted_continent = sorted(continent, key=lambda solution: solution.cost)


        print("best overall solution: ", sorted_continent[0].cost)
        print(time.time() - start_time)
        return sorted_continent[0].cost, global_average, sorted_continent[0]












