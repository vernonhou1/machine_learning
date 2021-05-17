import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from SVM import SVM


class LinearSeperableSVM(SVM):
    def __init__(self, file: str) -> None:
        SVM.__init__(self, file)

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

    def fit(self) -> List:
        column = len(self.df['x'])
        whole = []
        for i in range(column):
            tmp = []
            for j in range(column):
                tmp.append(self.df['label'][i] *
                           self.df['label'][j] *
                           (np.array([self.df['x'][i],
                                      self.df['y'][i]])
                            @ np.array([self.df['x'][j],
                                        self.df['y'][j]])))
            whole.append(tmp)
        whole = np.array(whole)
        s = np.diag(whole)
        P = matrix(whole, (column, column), 'd')
        q = np.full((column, 1), -1)
        q = matrix(q, (column, 1), 'd')
        G = np.zeros((column, column))
        row, col = np.diag_indices_from(G)
        G[row, col] = -1
        G = matrix(G, (column, column), 'd')
        h = np.full((column, 1), 0)
        h = matrix(h, (column, 1), 'd')
        b = matrix([0.0])
        A = matrix(self.df['label'], (1, column), 'd')
        sol = solvers.qp(P, q, G, h, A, b)

        w = 0
        b_right = 0
        for i in range(column):
            w += sol['x'][i]*self.df['label'][i] * \
                np.matrix(self.df['x'][i], self.df['y'][i])
            b_right += sol['x'][i]*self.df['label'][i] * \
                (np.mat(self.df['x'][i], self.df['y'][i]).transpose() @
                 np.mat(self.df['x'][0], self.df['y'][0]))
        b = self.df['label'][0] - b_right

        return [w.getA()[0], b.getA()[0]]

    @ staticmethod
    def predict(data: list, model: List) -> int:
        end = data[0]*model[0]+data[0]*model[0]-model[1]
        print(end)
        return end


if __name__ == "__main__":
    svm = LinearSeperableSVM("SVM/data.csv")
    model_1 = svm.fit()
    print(model_1)
    svm.predict_csv_visualization("SVM/data.csv", model_1)
