# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:28:42 2025

@author: jacks
"""

import numpy as np
import math

def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

# 給定數據點
x_data = np.array([0.698, 0.733, 0.768, 0.803])  # x
y_data = np.array([0.7661, 0.7432, 0.7193, 0.6946])  # f(x)

# 插值點
x_interp = 0.750  
real_value = math.cos(x_interp)  # 真實 cos(0.750)
print(f"真實值: cos({x_interp}) = {real_value}")

# 計算不同階數的 Lagrange 插值
for degree in range(1, min(4, len(x_data))):  # 確保 degree 不超過點的數量
    # 選擇最接近 x_interp 的 degree+1 個點
    idx = np.argsort(abs(x_data - x_interp))[:degree+1]
    x_subset = x_data[idx]
    #print(x_subset)
    y_subset = y_data[idx]
    
    # 計算 Lagrange 插值
    approx_value = lagrange_interpolation(x_subset, y_subset, x_interp)
    
    # 計算誤差界限
    max_derivative = 1  # cos(x) 的導數最大值為 1
    error_bound = abs(max_derivative / math.factorial(degree+1) * np.prod([x_interp - x for x in x_subset]))

    # 顯示結果
    print(f"Degree {degree}: Approximated cos({x_interp}) = {approx_value}, Error bound = {error_bound}")
