# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:27:09 2025

@author: jacks
"""

import math
import numpy as np

# === Define domain and step sizes ===
x_min, x_max = 0.0, math.pi
y_min, y_max = 0.0, math.pi / 2
h = k = 0.1 * math.pi

# Number of internal points in x and y directions
n = int((x_max - x_min) / h - 1)
m = int((y_max - y_min) / k - 1)

alpha = (h / k) ** 2

# === Generate grid points including boundaries ===
x = [x_min + i * h for i in range(n + 2)]  # i = 0 to n+1
y = [y_min + j * k for j in range(m + 2)]  # j = 0 to m+1

# === Initialize matrix A, vector F (RHS), and solution vector U ===
size = n * m  # number of unknowns
A = [[0.0 for _ in range(size)] for _ in range(size)]
F = [0.0 for _ in range(size)]
U = [0.0 for _ in range(size)]

# === Build system matrix A and right-hand side F ===
for i in range(1, n + 1):
    for j in range(1, m + 1):
        l = (i - 1) + n * (j - 1)  # map 2D (i,j) to 1D index
        xi = x[i]
        yj = y[j]

        # Fill matrix A using 5-point stencil
        if j > 1: A[l][l - n] = 1.0         # u_{i,j-1}
        if i > 1: A[l][l - 1] = alpha       # u_{i-1,j}
        A[l][l] = -2 * alpha - 2            # u_{i,j}
        if i < n: A[l][l + 1] = alpha       # u_{i+1,j}
        if j < m: A[l][l + n] = 1.0         # u_{i,j+1}

        # Compute F using source term and boundary adjustments
        F[l] = h**2 * xi * yj
        if j == 1: F[l] -= math.cos(xi)     # bottom boundary (u_{i,0})
        if i == 1: F[l] -= math.cos(yj)     # left boundary (u_{0,j})
        if i == n: F[l] -= -math.cos(yj)    # right boundary (u_{n+1,j})

# === Gaussian Elimination (Forward elimination) ===
for i in range(size):
    # Pivoting to improve numerical stability
    max_row = max(range(i, size), key=lambda r: abs(A[r][i]))
    A[i], A[max_row] = A[max_row], A[i]
    F[i], F[max_row] = F[max_row], F[i]

    # Eliminate variables
    for k in range(i + 1, size):
        if A[i][i] == 0:
            raise ValueError("No Solution")
        factor = A[k][i] / A[i][i]
        for j in range(i, size):
            A[k][j] -= factor * A[i][j]
        F[k] -= factor * F[i]

# === Back substitution ===
for i in range(size - 1, -1, -1):
    if A[i][i] == 0:
        raise ValueError("No Solution")
    U[i] = F[i]
    for j in range(i + 1, size):
        U[i] -= A[i][j] * U[j]
    U[i] /= A[i][i]

# === Map 1D solution vector U to 2D solution grid u[i][j] ===
u = [[0.0 for _ in range(m + 2)] for _ in range(n + 2)]
for i in range(1, n + 1):
    for j in range(1, m + 1):
        l = (i - 1) + n * (j - 1)
        u[i][j] = U[l]

# === Apply boundary conditions ===
for j in range(m + 2):
    u[0][j] = math.cos(y[j])        # Left boundary
    u[n + 1][j] = -math.cos(y[j])   # Right boundary
for i in range(n + 2):
    u[i][0] = math.cos(x[i])        # Bottom boundary
    u[i][m + 1] = 0.0               # Top boundary

# === Output the results ===
print("x\t y\t u(x,y)")
for j in range(m + 2):
    for i in range(n + 2):
        print(f"{x[i]:.3f}\t{y[j]:.3f}\t{u[i][j]:.6f}")
