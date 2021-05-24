import os
from numpy.core.fromnumeric import transpose
import pandas as pd
import numpy as np
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import math
from SVM_INIT import SVM_INIT

ModelType = List, float


class NonLinearSVM(SVM_INIT):
    def __init__(self,
                 file: str,
                 C: float,
                 kernel: str = "linear",
                 sigma: float = 0.0) -> None:
        SVM_INIT.__init__(self, file)
        self.c = C
        self.kernel = kernel
        self.sigma = sigma

    def fit(self) -> ModelType:
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
        G = matrix(G, (2 * column, column), 'd')
        h = np.vstack((h_1, h_2))
        h = matrix(h, (2 * column, 1), 'd')
        b = matrix([0.0])
        A = matrix(self.df['label'].values, (1, column), 'd')
        sol = solvers.qp(P, q, G, h, A, b)
        b_right = 0
        for i in range(column):
            if (self.kernel == "linear"):
                b_right += sol['x'][i]*self.df['label'][i] * \
                    (np.array([self.df['x'][i], self.df['y'][i]]).transpose() @
                    np.array([self.df['x'][0], self.df['y'][0]]))
            elif (self.kernel == "rbf"):
                b_right += sol['x'][i]*self.df['label'][i] * \
                    self.rbf(np.array([self.df['x'][i], self.df['y'][i]]),
                np.array([self.df['x'][0], self.df['y'][0]]),self.sigma)
            elif (self.kernel == "poly"):
                b_right += sol['x'][i]*self.df['label'][i] * \
                self.poly(np.array([self.df['x'][i], self.df['y'][i]]),
                np.array([self.df['x'][0], self.df['y'][0]]),self.sigma)
        b = self.df['label'][0] - b_right
        return sol['x'], b

    def compute_kernel(self, column: int) -> list:
        print(f'{self.kernel} is using...')
        whole = []
        if (self.kernel == "linear"):
            for i in range(column):
                tmp = []
                for j in range(column):
                    tmp.append(
                        self.df['label'][i] * self.df['label'][j] *
                        (np.array([self.df['x'][i], self.df['y'][i]])
                         @ np.array([self.df['x'][j], self.df['y'][j]])))
                whole.append(tmp)
        elif (self.kernel == "rbf"):
            for i in range(column):
                tmp = []
                for j in range(column):
                    tmp.append(
                        self.df['label'][i] * self.df['label'][j] *
                        self.rbf(np.array([self.df['x'][i], self.df['y'][i]]),
                                 np.array([self.df['x'][j], self.df['y'][j]]),
                                 self.sigma))
                whole.append(tmp)
        elif (self.kernel == "poly"):
            for i in range(column):
                tmp = []
                for j in range(column):
                    tmp.append(
                        self.df['label'][i] * self.df['label'][j] *
                        self.poly(np.array([self.df['x'][i], self.df['y'][i]]),
                                  np.array([self.df['x'][j], self.df['y'][j]]),
                                  self.sigma))
                whole.append(tmp)
        return whole

    @staticmethod
    def rbf(a: np.array, b: np.array, sigma: float) -> float:
        return math.exp(-((np.linalg.norm(a - b, ord=None))**2) / (2 * sigma))

    @staticmethod
    def poly(a: np.array, b: np.array, sigma: float) -> float:
        # print(f'a: {a},b: {b},poly: {(a @ b.transpose() + 1)**sigma}')
        return (a @ b.transpose() + 1)**sigma

    def predict(self, data: np.array, model: ModelType) -> int:
        result = 0
        for i in range(len(self.df['x'])):
            if (self.kernel == "linear"):
                result += model[0][i] * self.df['y'][i] * (np.array(
                    [self.df['x'][i], self.df['y'][i]]) @ data.transpose())
            elif (self.kernel == "rbf"):
                result += model[0][i] * self.df['y'][i] * self.rbf(
                    np.array([self.df['x'][i], self.df['y'][i]]), data,
                    self.sigma)
            elif (self.kernel == "poly"):
                result += model[0][i] * self.df['y'][i] * self.poly(
                    np.array([self.df['x'][i], self.df['y'][i]]), data,
                    self.sigma)
        return np.sign(result + model[1])


if __name__ == '__main__':
    # svm = NonLinearSVM("SVM/data.csv", 12, "rbf", 12)
    # svm = NonLinearSVM("SVM/data.csv", 0.001, "linear")
    svm = NonLinearSVM("SVM/data.csv", 6, "poly", 7.0)
    model_1 = svm.fit()
    df = pd.read_csv("SVM/test.csv")
    correct = 0
    for i in range(len(df)):
        ans = svm.predict(np.array([df['x'][i], df['y'][i]]), model_1)
        if df['label'][i] == ans:
            correct += 1
    print(f"accurary: {correct/len(df)}")
