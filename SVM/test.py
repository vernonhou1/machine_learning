import numpy as np
from cvxopt import solvers
from cvxopt import matrix
import matplotlib
import matplotlib.pyplot as plt

a = [[[3.0, 3.0], 1.0], [[4.0, 3.0], 1.0], [[1.0, 1.0], -1.0]]

fea = []
label = []
for i in a:
    fea.append(i[0])
    label.append(i[1])
column = len(label)
fea = np.array(fea)
label = np.array(label)


whole = []
for i in range(column):
    tmp = []
    for j in range(column):
        tmp.append(label[i]*label[j]*(fea[i]@fea[j]))
    whole.append(tmp)

whole = np.array(whole)
s = np.diag(whole)
P = matrix(whole, (column, column), 'd')
q = np.full((column, 1), -1)
q = matrix(q, (column, 1), 'd')
G = matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
h = np.full((column, 1), 0)
h = matrix(h, (column, 1), 'd')
b = matrix([0.0])
A = matrix(label, (1, column), 'd')
sol = solvers.qp(P, q, G, h, A, b)
