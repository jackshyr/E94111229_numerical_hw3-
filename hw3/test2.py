# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:50:05 2025

@author: jacks
"""

import numpy as np
import math

def lagrange_inverse_interpolation(fx_values, x_values, f_target):
    n = len(x_values)
    x_target = 0  # 目標值
    
    for i in range(n):
        term = x_values[i]
        for j in range(n):
            if i != j:
                term *= (f_target - fx_values[j]) / (fx_values[i] - fx_values[j])
        x_target += term
    
    return x_target

# 給定數據點 (x, f(x))，其中 f(x) = x - e^(-x)
x_values = np.array([0.3, 0.4, 0.5, 0.6])
y_values = np.array([0.740818, 
 0.670320,0.606531, 0.548812])
f_values = x_values - y_values  # 計算 f(x)
#print(f_values)

y_target = 0  # 目標值 (x - e^(-x) = 0)

# 使用 Lagrange 逆插值法
x_approx = lagrange_inverse_interpolation(f_values, x_values, y_target)

print(f"when x - e^(-x) = 0 時，x ≈ {x_approx:.6f}")
print(f"real e^-x={math.exp(-x_approx)}")