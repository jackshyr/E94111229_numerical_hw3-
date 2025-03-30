# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 19:46:07 2025

@author: jacks
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator, CubicHermiteSpline

def hermite_interpolation(t_values, d_values, v_values, t_target):
    hermite_spline = CubicHermiteSpline(t_values, d_values, v_values)
    d_pred = hermite_spline(t_target)
    v_pred = hermite_spline.derivative()(t_target)
    return d_pred, v_pred

def hermite_interpolation2(t_values, d_values, v_values, t_target):
    """
    自己建hermite
    """
    n = len(t_values)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))
    
    # 填充 z，將每個 t_values 重複一次
    z[::2] = t_values
    z[1::2] = t_values
    
    # 填充 Q 矩陣的第一列 (函數值)
    Q[::2, 0] = d_values
    Q[1::2, 0] = d_values
    
    # 填充 Q 矩陣的第二列 (導數值)
    Q[1::2, 1] = v_values
    Q[::2, 1] = v_values
    
    # 填充 Q 矩陣的高階差分
    for i in range(2, 2 * n):
        for j in range(2, i + 1):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])
    
    # 計算 Hermite 插值多項式的值
    t_diff = 1
    d_pred = Q[0, 0]
    for i in range(1, 2 * n):
        t_diff *= (t_target - z[i - 1])
        d_pred += Q[i, i] * t_diff
    
    # 計算 Hermite 插值多項式的一階導數值
    v_pred = Q[1, 1]
    t_diff = 1
    for i in range(2, 2 * n):
        t_term = 0
        for j in range(i):
            prod = 1
            for k in range(i):
                if k != j:
                    prod *= (t_target - z[k])
            t_term += prod
        v_pred += Q[i, i] * t_term
    
    return d_pred, v_pred
def find_exceed_speed_time(t_values, d_values, v_values, speed_limit):
    for t in np.linspace(min(t_values), max(t_values), 1000):
        _, v = hermite_interpolation(t_values, d_values, v_values, t)
        if v > speed_limit:
            return t
    return None

def find_max_speed(t_values, d_values, v_values):
    t_fine = np.linspace(min(t_values), max(t_values), 1000)
    max_speed = max(hermite_interpolation(t_values, d_values, v_values, t)[1] for t in t_fine)
    return max_speed

def find_exceed_speed_time2(t_values, d_values, v_values, speed_limit):
    for t in np.linspace(min(t_values), max(t_values), 1000):
        _, v = hermite_interpolation2(t_values, d_values, v_values, t)
        if v > speed_limit:
            return t
    return None

def find_max_speed2(t_values, d_values, v_values):
    t_fine = np.linspace(min(t_values), max(t_values), 1000)
    max_speed = max(hermite_interpolation2(t_values, d_values, v_values, t)[1] for t in t_fine)
    return max_speed

T = np.array([0, 3, 5, 8, 13])
D = np.array([0, 200, 375, 620, 990])
V = np.array([75, 77, 80, 74, 72])

t_target = 10
d_pred, v_pred = hermite_interpolation(T, D, V, t_target)
print("用python內建跑Hermite 插值法結果")
print(f"當 t = {t_target} 秒時，預測位置 D ≈ {d_pred:.2f} 英尺，速度 V ≈ {v_pred:.2f} 英尺/秒")


speed_limit = 80.67
t_exceed = find_exceed_speed_time(T, D, V, speed_limit)
if t_exceed:
    print(f"車輛首次超過 55 mi/h 的時間為 t ≈ {t_exceed:.2f} 秒")
else:
    print("車輛從未超過 55 mi/h 的速度限制")

max_speed = find_max_speed(T, D, V)
print(f"車輛的預測最大速度為 V_max ≈ {max_speed:.2f} 英尺/秒")
print("")
d_pred, v_pred = hermite_interpolation2(T, D, V, t_target)
print("newton建Hermite 插值法結果")
print(f"當 t = {t_target} 秒時，預測位置 D ≈ {d_pred:.2f} 英尺，速度 V ≈ {v_pred:.2f} 英尺/秒")

t_exceed = find_exceed_speed_time2(T, D, V, speed_limit)
if t_exceed:
    print(f"車輛首次超過 55 mi/h 的時間為 t ≈ {t_exceed:.2f} 秒")
else:
    print("車輛從未超過 55 mi/h 的速度限制")

max_speed = find_max_speed2(T, D, V)
print(f"車輛的預測最大速度為 V_max ≈ {max_speed:.2f} 英尺/秒")