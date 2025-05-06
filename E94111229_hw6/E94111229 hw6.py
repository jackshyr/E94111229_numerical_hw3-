# -*- coding: utf-8 -*-
"""
Created on Tue May  6 19:30:03 2025

@author: jacks
"""

import numpy as np

def gauss_elimination_with_partial_pivoting(A, b):
    n = len(b)
    for i in range(n):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

A1 = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
], dtype=float)

b1 = np.array([1.12, 3.44, 2.15, 4.16], dtype=float)

x1 = gauss_elimination_with_partial_pivoting(A1.copy(), b1.copy())
print("Problem 1 Solution x =", x1)
A2 = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

A2_inv = np.linalg.inv(A2)
print("Problem 2 Inverse of A =\n", A2_inv)
def crout_tridiagonal_solver(a, b, c, d):
    n = len(d)
    l = np.zeros(n)
    u = np.zeros(n-1)
    y = np.zeros(n)
    x = np.zeros(n)

    l[0] = b[0]
    for i in range(1, n):
        u[i-1] = a[i-1] / l[i-1]
        l[i] = b[i] - u[i-1] * c[i-1]

    y[0] = d[0]
    for i in range(1, n):
        y[i] = d[i] - u[i-1] * y[i-1]

    x[-1] = y[-1] / l[-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - c[i] * x[i+1]) / l[i]
    return x

# a: 下對角, b: 主對角, c: 上對角
a = [-1, -1, -1]
b = [3, 3, 3, 3]
c = [-1, -1, -1]
d = [2, 3, 4, 1]

x3 = crout_tridiagonal_solver(a, b, c, d)
print("Problem 3 Solution x =", x3)
