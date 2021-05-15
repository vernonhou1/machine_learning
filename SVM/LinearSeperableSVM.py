import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


class LinearSeperableSVM(object):
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

    def max_margin_method(self) -> List:
        column = len(self.df['x'])
        P = matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        q = matrix([0.0, 0.0, 0.0])
        df_xy = self.df.loc[:, 'x':'y'].multiply(
            self.df['label']*-1.0, axis="index")
        df_label = self.df.loc[:, 'label'].multiply(
            -1.0, axis="index")

        G = pd.concat([df_xy, df_label], axis=1).values
        G = matrix(G, (column, 3), 'd')
        h = np.full((column, 1), -1)
        h = matrix(h, (column, 1), 'd')
        result = solvers.qp(P, q, G, h)
        return result['x']

    @staticmethod
    def predict(data: list, model: List) -> int:
        end = data[0]*model[0]+data[0]*model[1]-model[2]
        print(end)
        return 1 if end > 6.5 else -1

    @staticmethod
    def predict_csv_visualization(file: str, model: List) -> None:
        df = pd.read_csv(file)
        sns.set()
        color_dic = {'1': 'r', '-1': 'b'}
        for i in range(0, len(df)):
            plt.plot(df['x'][i],
                     df['y'][i],
                     'o',
                     color=color_dic[f"{LinearSeperableSVM.predict([df['x'][i],df['y'][i]], model)}"])
        plt.show()


if __name__ == "__main__":
    svm = LinearSeperableSVM("SVM/data.csv")
    # svm.data_visualise()
    model = svm.max_margin_method()
    print(model)
    LinearSeperableSVM.predict_csv_visualization("SVM/data.csv", model)
