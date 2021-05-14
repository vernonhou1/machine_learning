
import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt


class LinearSeperableSVM(object):
    def __init__(self, file: str) -> None:
        super().__init__()
        self.df = pd.read_csv(file)

    def data_visualise(self) -> None:
        # print(self.df)
        sns.set()
        color_dic = {'1': 'r', '-1': 'b'}
        for i in range(0, len(self.df)):
            plt.plot(self.df['x'][i], self.df['y'][i], 'o',
                     color=color_dic[f"{self.df['label'][i]}"])
        plt.show()

    # def 


if __name__ == "__main__":
    svm = LinearSeperableSVM(
        "/home/vernonhou/pytorch_ws/machine-learning-in-action/SVM/data.csv")
    svm.data_visualise()
