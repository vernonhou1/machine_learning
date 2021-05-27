import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PCA(object):
    def __init__(self, data: np.array, k: int) -> None:
        super().__init__()
        self.data = data
        self.k = k

    def fit(self) -> np.array:
        data_mean = self.data.mean()
        data_len = len(self.data)
        data_array = np.zeros((self.data.shape[1], self.data.shape[1]))
        for i in range(data_len):
            data_array += (self.data[i] - data_mean)[:, None] @ (
                self.data[i] - data_mean)[:, None].T
        eigen, eigen_value = np.linalg.eig(data_array)
        eigen_dic = dict(zip(eigen, eigen_value))
        eigen_sorted = np.sort(eigen)
        A = data_array = np.zeros((self.k, self.data.shape[1]), dtype=complex)
        for i in range(self.k):
            A[i] = eigen_dic[eigen_sorted[i]]
        A /= np.linalg.norm(A)
        result = np.zeros((data_len, self.k), dtype=complex)
        for i in range(data_len):
            result[i] = A @ (self.data[i] - data_mean)[:, None]
        return result


if __name__ == '__main__':
    df = pd.read_csv('SVM/data.csv')[['x', 'y']].values
    pca = PCA(df, 1)
    result = pca.fit()
    plt.scatter(result, np.zeros((1, len(result))))
    plt.show()
