import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


class SVM_INIT(object):
    def __init__(self, file: str) -> None:
        super().__init__()
        self.df = pd.read_csv(file)

    def data_visualise(self) -> None:
        sns.set()
        color_dic = {'1': 'r', '-1': 'b'}
        for i in range(0, len(self.df)):
            plt.plot(self.df['x'][i],
                     self.df['y'][i],
                     'o',
                     color=color_dic[f"{self.df['label'][i]}"])
        plt.show()
        return None

    @ staticmethod
    def predict_csv_visualization(file: str, model: List) -> None:
        df = pd.read_csv(file)
        sns.set()
        color_dic = {'1': 'r', '-1': 'b'}
        for i in range(0, len(df)):
            plt.plot(df['x'][i],
                     df['y'][i],
                     'o',
                     color=color_dic[f"{df['label'][i]}"])
        aa = [0, 10]
        bb = [0*model[0]+model[1], 10*model[0]+model[1]]
        print(aa, bb)
        plt.plot(aa, bb, c="orange")
        plt.show()

    @ staticmethod
    def predict(data: list, model: List) -> int:
        end = data[0]*model[0]+data[0]*model[0]-model[1]
        return end
