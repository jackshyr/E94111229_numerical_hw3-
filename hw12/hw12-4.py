# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:31:40 2025

@author: jacks
"""

import numpy as np
import pandas as pd

# === 數值參數 ===
dx = 0.1
dt = 0.1
x_vals = np.arange(0, 1 + dx, dx)
t_vals = np.arange(0, 1 + dt, dt)
nx = len(x_vals)
nt = len(t_vals)

# 穩定性參數 (CFL): lambda^2 = (dt/dx)^2，這裡 = 1
lambda_sq = (dt / dx) ** 2

# === 建立解矩陣 p[n, i] ===
# n 是時間 index, i 是空間 index
p = np.zeros((nt, nx))

# === 初始條件 ===
p[0, :] = np.cos(2 * np.pi * x_vals)  # p(x, 0)

# === 初始速度條件：利用 Taylor 展開近似 t=dt 的值 ===
dpdt = 2 * np.pi * np.sin(2 * np.pi * x_vals)
for i in range(1, nx - 1):
    d2pdx2 = (p[0, i + 1] - 2 * p[0, i] + p[0, i - 1]) / dx ** 2
    p[1, i] = p[0, i] + dt * dpdt[i] + 0.5 * dt**2 * d2pdx2

# 邊界條件 at t = dt
p[1, 0] = 1
p[1, -1] = 2

# === 時間迭代：中心差分法 ===
for n in range(1, nt - 1):
    for i in range(1, nx - 1):
        p[n + 1, i] = 2 * p[n, i] - p[n - 1, i] + lambda_sq * (
            p[n, i + 1] - 2 * p[n, i] + p[n, i - 1]
        )
    # 套用邊界條件
    p[n + 1, 0] = 1
    p[n + 1, -1] = 2

# === 轉為 DataFrame 方便顯示或匯出 ===
df = pd.DataFrame(p, index=np.round(t_vals, 2), columns=np.round(x_vals, 2))
df.index.name = "t ↓"
df.columns.name = "x →"

# === 顯示結果（或 df.to_csv('wave_result.csv') 匯出）===
print(df)