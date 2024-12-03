import numpy as np

import Reporter
from Solution import Solution

"""
TODO:
- Make code more efficient, can use time to benchmark efficiency.
- accessing local variables is typically faster than accessing atributes!
  --> so move field attributes to local variables in relevant functions when possible
-
"""

class R0:

    """Setup of variables for the algorithm    """
    def __init__(self, filepath: str):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        with open(filepath) as file:
            self.distance_matrix: np.ndarray = np.loadtxt(file, delimiter=',')
        self.length: int = len(self.distance_matrix)
        self.limit: int = 300  #Time limit
        self.lambdaa: int = 2 * self.length
        self.mu: int = self.lambdaa
        self.k: int = 3
        self.seed_population: np.ndarray = np.array([Solution(self.length) for _ in range(self.lambdaa)])

    """ The main evolutionary algorithm loop."""
    def optimize(self):
        return 0

    """ Initializes the population randomly."""
    def intialize_random(self) -> None:
        for solution in self.seed_population:
            np.random.shuffle(solution.path[1:])

    """ Performs k-tournament selection to select pairs of parents by repeated selection."""
    def select_k_tournament_rep(self) -> np.ndarray:
        selected = np.zeros((self.mu, 2), dtype=int)
        for i in range(self.mu):
            parent_1 = self.tournament_selection(1)
            parent_2 = self.tournament_selection(1)
            parents = np.concatenate((parent_1, parent_2))
            selected[i,:] = parents
        return selected

    """"""
    def crossover(self):
        print("NOT YET IMPLEMENTED")

    def mutate(self):






    # ================================
    # Helper functions
    # ================================
    """ Calculates the cost of functions from scratch."""
    def calculate_cost(self, solution: Solution) -> None:
        path = solution.path
        cost = np.sum(self.distance_matrix[path[:-1], path[1:]])
        cost += self.distance_matrix[solution.path[-1], 0]
        solution.cost = cost

    """ Performs k-tournament selection to select n parents from k samples."""
    def tournament_selection(self, n: int) -> np.ndarray:
        candidates = np.random.choice(self.lambdaa, self.k, replace=True)
        candidate_parents = np.array([self.seed_population[i] for i in candidates])
        sorted_parents = sorted(candidate_parents, key = lambda solution: solution.cost)
        return np.array(sorted_parents[:n])













