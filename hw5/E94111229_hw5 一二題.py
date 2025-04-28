# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:33:55 2025

@author: jacks
"""

import numpy as np

# 第一題：單變數微分方程
def f(t, y):
    return 1 + (y/t) + (y/t)**2

def exact_solution(t):
    return t * np.tan(np.log(t))

def df_dt(t, y):
    dft = (-y) / (t**2) + (-2 * y**2) / (t**3)
    dfy = (1/t) + (2*y)/(t**2)
    return dft + dfy * f(t, y)

def euler_method(f, t0, y0, h, t_end):
    t_values = [t0]
    y_values = [y0]
    t, y = t0, y0
    while t < t_end:
        y += h * f(t, y)
        t += h
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

def taylor2_method(f, df_dt, t0, y0, h, t_end):
    t_values = [t0]
    y_values = [y0]
    t, y = t0, y0
    while t < t_end:
        y += h * f(t, y) + (h**2/2) * df_dt(t, y)
        t += h
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

# 第二題：聯立微分方程
def system(t, u):
    u1, u2 = u
    du1 = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2 = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1, du2])

def u1_exact(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def u2_exact(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

def runge_kutta_4(system, t0, u0, h, t_end):
    t_values = [t0]
    u_values = [np.array(u0)]
    t = t0
    u = np.array(u0)
    while t < t_end:
        k1 = h * system(t, u)
        k2 = h * system(t + h/2, u + k1/2)
        k3 = h * system(t + h/2, u + k2/2)
        k4 = h * system(t + h, u + k3)
        u = u + (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
        t_values.append(t)
        u_values.append(u.copy())
    return np.array(t_values), np.array(u_values)

def relative_error(approx, exact):
    if exact == 0:
        return 0.0
    return abs((approx - exact) / exact)

# ========================== 主程式 ==========================

# 第一題參數
t0 = 1
y0 = 0
h = 0.1
t_end = 2

t_euler, y_euler = euler_method(f, t0, y0, h, t_end)
t_taylor, y_taylor = taylor2_method(f, df_dt, t0, y0, h, t_end)
y_exact_at_nodes = exact_solution(t_euler)

# 第一題輸出
print("第一題結果：")
print("t值       Euler近似值    Taylor近似值   真實值    Euler相對誤差   Taylor相對誤差")
for i in range(len(t_euler)):
    rel_err_euler = relative_error(y_euler[i], y_exact_at_nodes[i])
    rel_err_taylor = relative_error(y_taylor[i], y_exact_at_nodes[i])
    print(f" {t_euler[i]:.1f}      {y_euler[i]:.6f}      {y_taylor[i]:.6f}      {y_exact_at_nodes[i]:.6f}      {rel_err_euler:.6f}       {rel_err_taylor:.6f}")

# 第二題參數
t0_sys = 0
u0 = [4/3, 2/3]
h1 = 0.05
h2 = 0.1
t_end_sys = 1

# Runge-Kutta 方法
t_rk1, u_rk1 = runge_kutta_4(system, t0_sys, u0, h1, t_end_sys)
t_rk2, u_rk2 = runge_kutta_4(system, t0_sys, u0, h2, t_end_sys)

u1_exact_vals1 = u1_exact(t_rk1)
u2_exact_vals1 = u2_exact(t_rk1)


u1_exact_vals2 = u1_exact(t_rk2)
u2_exact_vals2 = u2_exact(t_rk2)


# 第二題輸出
print("\n第二題結果（Runge-Kutta h=0.05）：")
print(f"{'t值':>6} {'u1近似值':>9} {'u2近似值':>9} {'u1相對誤差':>9} {'u1真實值':>9} {'u2真實值':>9} {'u2相對誤差':>9}")
for i in range(len(t_rk1)):
    t_val = t_rk1[i]
    u1_approx = u_rk1[i, 0]
    u2_approx = u_rk1[i, 1]
    u1_exact_val = u1_exact_vals1[i]
    u2_exact_val = u2_exact_vals1[i]
    rel_err_u1 = relative_error(u1_approx, u1_exact_val)
    rel_err_u2 = relative_error(u2_approx, u2_exact_val)
    
    print(f"{t_val:6.2f} {u1_approx:12.6f} {u2_approx:12.6f} {rel_err_u1:12.6f} {u1_exact_val:12.6f} {u2_exact_val:12.6f} {rel_err_u2:12.6f}")

print("\n第二題結果（Runge-Kutta h=0.1）：")
print(f"{'t值':>8} {'u1近似值':>20} {'u2近似值':>20} {'u1相對誤差':>20} {'u1真實值':>9} {'u2真實值':>9} {'u2相對誤差':>20}")
for i in range(len(t_rk2)):
    t_val = t_rk2[i]
    u1_approx = u_rk2[i, 0]
    u2_approx = u_rk2[i, 1]
    u1_exact_val = u1_exact_vals2[i]
    u2_exact_val = u2_exact_vals2[i]
    rel_err_u1 = relative_error(u1_approx, u1_exact_val)
    rel_err_u2 = relative_error(u2_approx, u2_exact_val)
    print(f"{t_rk2[i]:8.2f} {u_rk2[i,0]:23.6f} {u_rk2[i,1]:23.6f} {rel_err_u1:23.6f} {u1_exact_vals2[i]:12.6f} {u2_exact_vals2[i]:12.6f} {rel_err_u2:23.6f}")

