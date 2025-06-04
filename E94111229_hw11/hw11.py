# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:39:14 2025

@author: jacks
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp

# 定義原始微分方程的右邊
def differential_equation(x, y, dy_dx):
    return -(x + 1) * dy_dx + 2 * y + (1 - x**2) * np.exp(-x)

# 方法一：Shooting Method
def shooting_method():
    print("METHOD 1: SHOOTING METHOD")

    h = 0.1
    x_span = np.arange(0, 1.01, h)  

    # 將二階微分方程轉換為一階系統
    def system_ode(t, y):
        y1, y2 = y  # y1 = y, y2 = y'
        dy1_dt = y2
        dy2_dt = -(t + 1) * y2 + 2 * y1 + (1 - t**2) * np.exp(-t)
        return [dy1_dt, dy2_dt]

    # 給定初始斜率，求解初值問題
    def solve_with_slope(slope):
        y0 = [1, slope]
        sol = solve_ivp(system_ode, [0, 1], y0, t_eval=x_span, method='RK45')
        return sol.y[0][-1]  # 回傳 y(1)

    # 目標函數: 使 y(1) 趨近於 2
    def objective(slope):
        return abs(solve_with_slope(slope) - 2)

    # 最小化目標函數，尋找最佳初始斜率
    result = minimize_scalar(objective, bounds=(-10, 10), method='bounded')
    optimal_slope = result.x

    # 用最佳斜率重新解 ODE
    y0 = [1, optimal_slope]
    sol = solve_ivp(system_ode, [0, 1], y0, t_eval=x_span, method='RK45')
    x_shoot = sol.t
    y_shoot = sol.y[0]
    dy_shoot = sol.y[1]

    # 輸出解
    print(f"Boundary conditions: y(0) = 1, y(1) = 2")
    print(f"Optimal initial slope: y'(0) = {optimal_slope:.6f}")
    print(f"Achieved final value: y(1) = {y_shoot[-1]:.6f}")
    print(f"Error in boundary condition: {abs(y_shoot[-1] - 2):.8f}")
    print("\nSolution points:")
    for i in range(len(x_shoot)):
        print(f"x = {x_shoot[i]:.1f}, y = {y_shoot[i]:.6f}, y' = {dy_shoot[i]:.6f}")

    return x_shoot, y_shoot, dy_shoot

# 方法二：有限差分法 Finite Difference
def finite_difference_method():
    print("METHOD 2: FINITE DIFFERENCE METHOD")

    h = 0.1
    n = int(1/h) + 1
    x = np.linspace(0, 1, n)

    # 建立係數矩陣 A 和常數項 b
    A = np.zeros((n, n))
    b = np.zeros(n)

    # 邊界條件
    A[0, 0] = 1
    b[0] = 1  # y(0) = 1
    A[-1, -1] = 1
    b[-1] = 2  # y(1) = 2

    # 中間點用差分法逼近
    for i in range(1, n-1):
        xi = x[i]
        A[i, i-1] = 1/h**2 + (xi + 1)/(2*h)           # y[i-1]
        A[i, i]   = -2/h**2 - 2                       # y[i]
        A[i, i+1] = 1/h**2 - (xi + 1)/(2*h)           # y[i+1]
        b[i] = -(1 - xi**2) * np.exp(-xi)

    # 解線性系統
    y_fd = np.linalg.solve(A, b)

    # 輸出解
    print(f"Grid points: {n}")
    print(f"Step size: h = {h}")
    print("Boundary conditions: y(0) = 1, y(1) = 2")
    print("\nSolution points:")
    for i in range(len(x)):
        print(f"x = {x[i]:.1f}, y = {y_fd[i]:.6f}")

    return x, y_fd

# 方法三：變分法 (Galerkin Variational Method)
def variational_method():
    print("METHOD 3: VARIATIONAL APPROACH (GALERKIN METHOD)")

    # 選取基底函數，滿足邊界條件
    # y(x) = 1 + x + Σ c_i * x^i * (1 - x)
    def basis_function(x, i):
        if i == 0:
            return 1 + x
        else:
            return x**i * (1 - x)

    def basis_derivative(x, i, order=1):
        if order == 1:
            if i == 0:
                return np.ones_like(x)
            else:
                return i * x**(i-1) * (1 - x) - x**i
        elif order == 2:
            if i == 0:
                return np.zeros_like(x)
            elif i == 1:
                return -2 * np.ones_like(x)
            else:
                return i * (i-1) * x**(i-2) * (1 - x) - 2 * i * x**(i-1)

    n_basis = 5  # 使用的基底函數數量
    x_quad = np.linspace(0, 1, 101)  # 積分點（數值積分用）

    A_var = np.zeros((n_basis-1, n_basis-1))
    b_var = np.zeros(n_basis-1)

    # Galerkin 條件：內積 <test_func, L[y] - f> = 0
    for i in range(n_basis-1):
        for j in range(n_basis-1):
            L_phi = (basis_derivative(x_quad, j+1, 2) +
                     (x_quad + 1) * basis_derivative(x_quad, j+1, 1) -
                     2 * basis_function(x_quad, j+1))
            integrand = basis_function(x_quad, i+1) * L_phi
            A_var[i, j] = np.trapz(integrand, x_quad)

        L_phi0 = (basis_derivative(x_quad, 0, 2) +
                  (x_quad + 1) * basis_derivative(x_quad, 0, 1) -
                  2 * basis_function(x_quad, 0))
        f_val = (1 - x_quad**2) * np.exp(-x_quad)
        rhs_integrand = basis_function(x_quad, i+1) * (f_val - L_phi0)
        b_var[i] = np.trapz(rhs_integrand, x_quad)

    # 解線性方程組得到係數 c_i
    c = np.linalg.solve(A_var, b_var)

    # 計算近似解
    x_var = np.linspace(0, 1, 11)
    y_var = basis_function(x_var, 0)
    for i in range(len(c)):
        y_var += c[i] * basis_function(x_var, i+1)

    print(f"Number of basis functions: {n_basis}")
    print(f"Coefficients for higher-order terms: {c}")
    print(f"Boundary conditions satisfied: y(0) = {y_var[0]:.6f}, y(1) = {y_var[-1]:.6f}")
    print("\nSolution points:")
    for i in range(len(x_var)):
        print(f"x = {x_var[i]:.1f}, y = {y_var[i]:.6f}")

    return x_var, y_var, c

# 主執行程式
def main():

    x1, y1, dy1 = shooting_method()
    x2, y2 = finite_difference_method()
    x3, y3, c = variational_method()

if __name__ == "__main__":
    main()