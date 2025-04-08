# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 20:54:04 2025

@author: jacks
"""

import numpy as np
import scipy.integrate as spi

# 題目一的函數
def f1(x):
    return np.exp(x) * np.sin(4 * x)

# 題目一的區間與步長
a1, b1 = 1, 2
h = 0.1

# 複合梯形法
def trapezoidal_rule(f, a, b, h):
    x = np.arange(a, b + h, h)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# 複合 Simpson’s 法
def simpsons_rule(f, a, b, h):
    if (b - a) / h % 2 != 0:
        raise ValueError("區間需為偶數以使用 Simpson's Rule")
    x = np.arange(a, b + h, h)
    y = f(x)
    return h / 3 * (y[0] + 2 * np.sum(y[2:-1:2]) + 4 * np.sum(y[1::2]) + y[-1])

# 複合 Midpoint 法
def midpoint_rule(f, a, b, h):
    n = int((b - a) / h) 
    midpoints = a + h * (np.arange(n) + 0.5)  # 使用中點公式生成 midpoints
    return h * np.sum(f(midpoints))

# 題目二的函數
def f2(x):
    return x**2 * np.log(x)

# 高斯積分通用公式（Legendre Polynomial）
def gaussian_quadrature(f, a, b, n):
    [x, w] = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (x + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w * f(t))

# 題目二精確值
exact_value_q2, _ = spi.quad(f2, 1, 1.5)

# 執行所有方法
print("== 題目一 ==")
print("複合梯形法結果:", trapezoidal_rule(f1, a1, b1, h))
print("複合 Simpson 法結果:", simpsons_rule(f1, a1, b1, h))
print("複合 Midpoint 法結果:", midpoint_rule(f1, a1, b1, h))

print("\n== 題目二 ==")
print("Gaussian Quadrature (n=3):", gaussian_quadrature(f2, 1, 1.5, 3))
print("Gaussian Quadrature (n=4):", gaussian_quadrature(f2, 1, 1.5, 4))
print("精確值:", exact_value_q2)
# 被積分函數 f(x, y)
def f3(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# Simpson's Rule for double integral
def simpsons_double_integral(f, ax, bx, ay_func, by_func, nx, ny):
    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("nx 和 ny 必須為偶數")
    
    hx = (bx - ax) / nx
    result = 0

    for i in range(nx + 1):
        x = ax + i * hx
        wx = 1 if i == 0 or i == nx else 4 if i % 2 != 0 else 2

        ay = ay_func(x)
        by = by_func(x)
        hy = (by - ay) / ny

        inner_sum = 0
        for j in range(ny + 1):
            y = ay + j * hy
            wy = 1 if j == 0 or j == ny else 4 if j % 2 != 0 else 2
            inner_sum += wy * f(x, y)

        result += wx * inner_sum * hy / 3

    return hx / 3 * result

# Gaussian Quadrature for double integral
def gaussian_double_integral(f, ax, bx, ay_func, by_func, nx, ny):
    [x_nodes, x_weights] = np.polynomial.legendre.leggauss(nx)
    [y_nodes, y_weights] = np.polynomial.legendre.leggauss(ny)

    result = 0
    for i in range(nx):
        x = 0.5 * (bx - ax) * x_nodes[i] + 0.5 * (bx + ax)
        wx = x_weights[i]
        ay = ay_func(x)
        by = by_func(x)

        for j in range(ny):
            y = 0.5 * (by - ay) * y_nodes[j] + 0.5 * (by + ay)
            wy = y_weights[j]
            result += wx * wy * f(x, y) * 0.25 * (bx - ax) * (by - ay)

    return result

# 精確值使用 scipy
def f3_exact(y, x):
    return 2 * y * np.sin(x) + np.cos(x)**2

# 設定積分範圍與方法參數
a, b = 0, np.pi / 4
nx = ny = 4  # Simpson
ng = 3       # Gaussian

# 計算值
simpson_result = simpsons_double_integral(f3, a, b, np.sin, np.cos, nx, ny)
gauss_result = gaussian_double_integral(f3, a, b, np.sin, np.cos, ng, ng)
exact_result, _ = spi.dblquad(f3_exact, a, b, lambda x: np.sin(x), lambda x: np.cos(x))

# 輸出結果
print("\n== 題目三 ==")
print("Simpson’s Rule Approximation:", simpson_result)
print("Gaussian Quadrature Approximation:", gauss_result)
print("Exact Value (dblquad):", exact_result)
# Composite Simpson's Rule
def composite_simpson(f, a, b, n):
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's rule.")
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h/3 * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])

# Part (a): ∫₀¹ x^(-1/4) * sin(x) dx
def f_a(x):
    return np.where(x == 0, 0, x**(-1/4) * np.sin(x))

# Part (b): ∫₁^∞ x^(-4) * sin(x) dx
# After transformation: ∫₀¹ t^2 * sin(1/t) dt
def f_b(t):
    return np.where(t == 0, 0, t**2 * np.sin(1/t))

# Set n = 4
n = 4
a1, b1 = 1e-6, 1
# Compute the integrals
a_result = composite_simpson(f_a, a1, b1, n)
b_result = composite_simpson(f_b, 1e-5, 1, n)  # avoid t = 0
print("\n== 題目四 ==")
print("Approximation for part (a):", a_result)
print("Approximation for part (b):", b_result)