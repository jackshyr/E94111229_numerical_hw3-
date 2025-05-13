# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:48:38 2025

@author: jacks
"""

import numpy as np
from scipy.sparse.linalg import cg

# 系統矩陣與常數向量
A = np.array([
    [4, -1,  0, -1,  0,  0],
    [-1, 4, -1,  0, -1,  0],
    [0, -1, 4,  0,  1, -1],
    [-1, 0,  0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)


def jacobi(A, b, tol=1e-8, max_iter=1000):
    x = np.zeros_like(b)
    n = len(b)
    x_new = x.copy()
    for it in range(1, max_iter+1):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, it
        x[:] = x_new
    return x, max_iter

def gauss_seidel(A, b, tol=1e-8, max_iter=1000):
    x = np.zeros_like(b)
    n = len(b)
    for it in range(1, max_iter+1):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, it
        x[:] = x_new
    return x, max_iter

def sor(A, b, omega=1.1, tol=1e-8, max_iter=1000):
    x = np.zeros_like(b)
    n = len(b)
    for it in range(1, max_iter+1):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i+1, n))
            x[i] = (1-omega)*x_old[i] + omega*(b[i] - s1 - s2)/A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, it
    return x, max_iter

def conjugate_gradient(A, b, tol=1e-8, max_iter=1000):
    num_iters = [0]
    def callback(xk):
        num_iters[0] += 1
    x, info = cg(A, b, x0=np.zeros_like(b), rtol=tol, atol=0, maxiter=max_iter, callback=callback)
    return x, num_iters[0] if num_iters[0] > 0 else max_iter

# 執行四種法
x_jacobi, it_jacobi = jacobi(A, b)
x_gs, it_gs = gauss_seidel(A, b)
x_sor, it_sor = sor(A, b, omega=1.1)
x_cg, it_cg = conjugate_gradient(A, b, tol=1e-8, max_iter=1000)

# 結果輸出
np.set_printoptions(precision=7, suppress=True)
print(f"(a) Jacobi: {x_jacobi} (iterations: {it_jacobi})")
print(f"(b) Gauss-Seidel: {x_gs} (iterations: {it_gs})")
print(f"(c) SOR (ω=1.1): {x_sor} (iterations: {it_sor})")
print(f"(d) Conjugate Gradient: {x_cg} (iterations: {it_cg})")
