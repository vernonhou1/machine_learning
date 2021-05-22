import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import math
from SVM_INIT import SVM_INIT


class NonLinearSVM(SVM_INIT):

    def __init__(self, file: str, C: float, kernel: str = "linear", sigma: float = 0.0) -> None:
        SVM_INIT.__init__(self, file)
        self.c = C
        self.kernel = kernel
        self.sigma = sigma

    def fit(self) -> List:
        column = len(self.df['x'])
        whole = self.compute_kernel(column)
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

    def compute_kernel(self, column: int) -> list:
        print(f'{self.kernel} is using...')
        whole = []
        if (self.kernel == "linear"):
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
        elif (self.kernel == "rbf"):
            for i in range(column):
                tmp = []
                for j in range(column):
                    tmp.append(self.df['label'][i] *
                               self.df['label'][j] *
                               self.rbf(self.df['x'], self.df['y'], self.sigma))
                whole.append(tmp)
        elif (self.kernel == "poly"):
            for i in range(column):
                tmp = []
                for j in range(column):
                    tmp.append(self.df['label'][i] *
                               self.df['label'][j] *
                               self.poly(self.df['x'], self.df['y'], self.sigma))
                whole.append(tmp)
        return whole

    @staticmethod
    def rbf(a: list, b: list, sigma: float) -> float:
        return math.exp(-((np.linalg.norm(a-b, ord=None))**2)/(2*sigma))

    @staticmethod
    def poly(a: list, b: list, sigma: float) -> float:
        return (np.array(a)@np.array(b)+1)**sigma


if __name__ == '__main__':
    svm = NonLinearSVM("SVM/data.csv", 12, "poly", 0.5)
    # svm.data_visualise()
    model_1 = svm.fit()
    svm.predict_csv_visualization("SVM/data.csv", model_1)
