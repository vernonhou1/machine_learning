import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Perceptron(object):
    def __init__(self, data: np.array) -> None:
        super().__init__()
        self.data = data
        self.W = np.random.randn(self.data.shape[1])

    def fit(self) -> np.array:
        for i in range(self.data.shape[0]):
            if self.data[i][-1] == -1:
                self.data[i][:-1] *= -1
        while (True):
            for i in range(self.data.shape[0]):
                if (self.W.T @ self.data[i]) < 0:
                    self.W += self.data[i]
                else:
                    for i in range(self.data.shape[0]):
                        if (any(self.W.T @ _ < 0 for _ in self.data)):
                            break
                        else:
                            self.model = self.W
                            print(f'final: {self.W}')
                            return self.model

    def predict(self, data: np.array) -> int:
        return np.sign(self.model[:-1].T @ data + self.model[-1])


if __name__ == '__main__':
    df = pd.read_csv("SVM/data.csv")

    per = Perceptron(df.values)
    model = per.fit()
    test = pd.read_csv("SVM/test.csv")
    for i in range(len(test)):
        print(test.iloc[i].values)
        ans = per.predict(test.iloc[i].values[:-1])
        print(ans)
