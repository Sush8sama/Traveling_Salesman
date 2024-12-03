import numpy as np


class Solution:
    def __init__(self, length: int):
        self.path: np.ndarray = np.arange(length, dtype=int)
        self.cost: int = 0
        self.cost_correct: bool = False
        self.penalty: int = 0
        self.age: int = 0
        # self.params: np.ndarray = np.arange(params_length, dtype=int)









