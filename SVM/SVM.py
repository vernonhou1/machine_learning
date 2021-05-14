import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import random


class SVM(object):
    def __init__(self, file) -> None:
        super().__init__()
        self.df = pd.read_csv(file)

    def __select_jrand(self, i: int, m: int) -> int:
        j = i
        while (j == i):
            j = int(random.uniform(0, m))
        return j

    def __clipAlpha(self, aj: float, H: float, L: float) -> float:
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def simple_SMO(self, C: int, toler: float, maxIter: int) -> None:
        pass