# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:28:42 2025

@author: jacks
"""

import numpy as np
import scipy.interpolate as interp
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
x_data = np.array([0.698, 0.733, 0.750, 0.768, 0.803])
y_data = np.array([0.7661, 0.7432, 0.7317, 0.7193, 0.6946])
real_data = np.array([math.cos(0.698), math.cos(0.733),math.cos(0.750), math.cos(0.768), math.cos(0.803)])
print(real_data)
# 插值點
x_interp = 0.750

# 計算不同階數的 Lagrange 插值
for i in range(0,5):
    x_interp = x_data[i]
    print(real_data[i])
    for degree in range(1, 5):
        approx_value = lagrange_interpolation(x_data[:degree+1], y_data[:degree+1], x_interp)
        error = real_data[i]-approx_value
        #error_bound = abs(math.sin(x_interp)) / math.factorial(degree+1) * np.prod([x_interp - x for x in x_data[:degree+1]])
        print(f"Degree {degree}: Approximated cos({x_interp}) = {approx_value}, Error bound = {abs(error)}")
    print("\n")
