
import os
import pandas as pd
import numpy as np
from typing import List, Tuple


class LinearSeperableSVM(object):
    def __init__(self, file: str) -> None:
        super().__init__()
        self.df = pd.read_csv(file)

    def model(self, df: pd.DataFrame) -> None:
        pass
