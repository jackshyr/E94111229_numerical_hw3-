# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:31:39 2025

@author: jacks
"""

import math

# Domain boundaries in polar coordinates
r_min = 0.5
r_max = 1.0
theta_min = 0.0
theta_max = math.pi / 3

# Step sizes
h = 0.05                      # radial step
k = math.pi / 30             # angular step

# Grid sizes
n = int((r_max - r_min) / h - 1)
m = int((theta_max - theta_min) / k - 1)

alpha = (h / k) ** 2         # coefficient used in discretization

# Discrete coordinate values
r = [r_min + i * h for i in range(n + 2)]        # r[0..n+1]
theta = [theta_min + j * k for j in range(m + 2)]  # θ[0..m+1]

size = n * m   # Total number of unknowns

# Initialize coefficient matrix A, RHS vector F, and solution vector U
A = [[0.0 for _ in range(size)] for _ in range(size)]
F = [0.0 for _ in range(size)]
U = [0.0 for _ in range(size)]

# Build linear system A * U = F
for i in range(1, n + 1):           # radial index
    for j in range(1, m + 1):       # angular index
        idx = (j - 1) * n + (i - 1) # linear index for (i,j)
        ri = r[i]                   # current radius

        # Coefficients for neighbors
        c_theta_prev = alpha
        c_r_prev = ri**2 - (h / 2) * ri
        c_center = -2 * (alpha + ri**2)
        c_r_next = ri**2 + (h / 2) * ri
        c_theta_next = alpha

        # Fill matrix A with coefficients
        if j > 1:             # T_{i,j-1}
            A[idx][idx - n] = c_theta_prev
        if i > 1:             # T_{i-1,j}
            A[idx][idx - 1] = c_r_prev
        A[idx][idx] = c_center  # T_{i,j}
        if i < n:             # T_{i+1,j}
            A[idx][idx + 1] = c_r_next
        if j < m:             # T_{i,j+1}
            A[idx][idx + n] = c_theta_next

        # Fill RHS vector F based on boundary conditions
        if j == 1:     # θ = 0: T = 0
            F[idx] -= c_theta_prev * 0
        if i == 1:     # r = 0.5: T = 50
            F[idx] -= c_r_prev * 50
        if i == n:     # r = 1.0: T = 100
            F[idx] -= c_r_next * 100
        if j == m:     # θ = π/3: T = 0
            F[idx] -= c_theta_next * 0

# Solve the linear system using Gaussian Elimination
for i in range(size):
    # Partial pivoting
    max_row = max(range(i, size), key=lambda x: abs(A[x][i]))
    A[i], A[max_row] = A[max_row], A[i]
    F[i], F[max_row] = F[max_row], F[i]

    for k in range(i + 1, size):
        if A[i][i] == 0:
            raise ValueError("No Solution")
        factor = A[k][i] / A[i][i]
        for j in range(i, size):
            A[k][j] -= factor * A[i][j]
        F[k] -= factor * F[i]

# Back-substitution
for i in range(size - 1, -1, -1):
    if A[i][i] == 0:
        raise ValueError("No Solution")
    U[i] = F[i]
    for j in range(i + 1, size):
        U[i] -= A[i][j] * U[j]
    U[i] /= A[i][i]

# Construct 2D solution grid T[i][j]
T = [[0.0 for _ in range(m + 2)] for _ in range(n + 2)]
for i in range(1, n + 1):
    for j in range(1, m + 1):
        idx = (j - 1) * n + (i - 1)
        T[i][j] = U[idx]

# Apply boundary conditions
for j in range(m + 2):
    T[0][j] = 50.0       # r = 0.5
    T[n + 1][j] = 100.0  # r = 1.0
for i in range(n + 2):
    T[i][0] = 0.0        # θ = 0
    T[i][m + 1] = 0.0    # θ = π/3

# Output the result table
print("r\t θ\t T(r,θ)")
for j in range(m + 2):
    for i in range(n + 2):
        print(f"{r[i]:.3f}\t{theta[j]:.3f}\t{T[i][j]:.6f}")