import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from SVM_INIT import SVM_INIT


class LinearSVM(SVM_INIT):

    def __init__(self, file: str, C: float) -> None:
        SVM_INIT.__init__(self, file)
        self.c = C

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
        G_1 = np.zeros((column, column))
        row, col = np.diag_indices_from(G_1)
        G_1[row, col] = -1
        G_1 = matrix(G_1, (column, column), 'd')
        G_2 = np.zeros((column, column))
        G_2[row, col] = 1
        G = np.vstack((G_1, G_2))
        h_1 = np.full((column, 1), 0)
        h_2 = np.full((column, 1), self.c)
        G = matrix(G, (2*column, column), 'd')
        h = np.vstack((h_1, h_2))
        h = matrix(h, (2*column, 1), 'd')
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


if __name__ == '__main__':
    svm = LinearSVM("SVM/data.csv", 0.4)
    # svm.data_visualise()
    model_1 = svm.fit()
    print(model_1)
    svm.predict_csv_visualization("SVM/data.csv", model_1)
