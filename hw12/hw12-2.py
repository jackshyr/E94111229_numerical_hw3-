# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:31:17 2025

@author: jacks
"""

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# === 基本參數 ===
dr = 0.1
dt = 0.5
K = 0.1
alpha = 1 / (4 * K)

# === 使用 linspace 確保正確格點 ===
r_vals = np.linspace(0.5, 1.0, 6)     # Δr = 0.1 → [0.5, 0.6, ..., 1.0]
t_vals = np.linspace(0, 10, 21)       # Δt = 0.5 → [0, 0.5, ..., 10]
nr = len(r_vals)
nt = len(t_vals)

# === 初始化三種方法的解矩陣 ===
T_f = np.zeros((nt, nr))  # Forward Difference
T_b = np.zeros((nt, nr))  # Backward Difference
T_c = np.zeros((nt, nr))  # Crank-Nicolson

# === 初始條件與邊界條件 ===
T_f[0, :] = T_b[0, :] = T_c[0, :] = 200 * (r_vals - 0.5)
T_f[:, -1] = T_b[:, -1] = T_c[:, -1] = 100 + 40 * t_vals

# === (a) Forward Difference Method ===
for n in range(0, nt - 1):
    for i in range(1, nr - 1):
        r = r_vals[i]
        T_f[n+1, i] = T_f[n, i] + alpha * dt * (
            (T_f[n, i+1] - 2*T_f[n, i] + T_f[n, i-1]) / dr**2 +
            (1/r) * (T_f[n, i+1] - T_f[n, i-1]) / (2*dr)
        )
    T_f[n+1, 0] = T_f[n+1, 1] / (1 + 3 * dr)

# === (b) Backward & (c) Crank-Nicolson Methods ===
for method, T in [('b', T_b), ('c', T_c)]:
    for n in range(0, nt - 1):
        T_prev = T[n, :]
        main_diag = np.zeros(nr - 2)
        upper_diag = np.zeros(nr - 3)
        lower_diag = np.zeros(nr - 3)
        rhs = np.zeros(nr - 2)

        for i in range(1, nr - 1):
            r = r_vals[i]
            lam_r = alpha * dt / dr**2
            gamma = alpha * dt / (2 * r * dr)
            j = i - 1

            if method == 'b':
                main_diag[j] = 1 + 2 * lam_r
                if i != 1:
                    lower_diag[j - 1] = -lam_r - gamma
                if i != nr - 2:
                    upper_diag[j] = -lam_r + gamma
                rhs[j] = T_prev[i]

            elif method == 'c':
                main_diag[j] = 1 + lam_r
                if i != 1:
                    lower_diag[j - 1] = -0.5 * (lam_r + gamma)
                if i != nr - 2:
                    upper_diag[j] = -0.5 * (lam_r - gamma)
                rhs[j] = (
                    (1 - lam_r) * T_prev[i] +
                    (0.5 * (lam_r + gamma) * T_prev[i - 1] if i > 1 else 0) +
                    (0.5 * (lam_r - gamma) * T_prev[i + 1] if i < nr - 2 else 0)
                )

        A = diags([main_diag, lower_diag, upper_diag], [0, -1, 1])
        T_interior = spsolve(A.tocsr(), rhs)
        T[n+1, 1:-1] = T_interior
        T[n+1, 0] = T[n+1, 1] / (1 + 3 * dr)

# === 輸出 DataFrame（全表格）===
df_f = pd.DataFrame(T_f, index=np.round(t_vals, 2), columns=np.round(r_vals, 2))
df_b = pd.DataFrame(T_b, index=np.round(t_vals, 2), columns=np.round(r_vals, 2))
df_c = pd.DataFrame(T_c, index=np.round(t_vals, 2), columns=np.round(r_vals, 2))

df_f.index.name = df_b.index.name = df_c.index.name = "t ↓"
df_f.columns.name = df_b.columns.name = df_c.columns.name = "r →"

# === 顯示所有結果 ===
print("\n==== (a) Forward Difference ====\n")
print(df_f)

print("\n==== (b) Backward Difference ====\n")
print(df_b)

print("\n==== (c) Crank-Nicolson ====\n")
print(df_c)