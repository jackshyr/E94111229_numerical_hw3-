# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:50:28 2025

@author: jacks
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad

# ========== 題目1 ==========
print("題目1：最小二乘法近似")
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# (a) 二次多項式
A = np.vstack([x**2, x, np.ones(len(x))]).T
c_poly, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
y_poly = c_poly[0]*x**2 + c_poly[1]*x + c_poly[2]
err_poly = np.sum((y - y_poly)**2)
print(f"(a) y = {c_poly[0]:.12f}x^2 + {c_poly[1]:.12f}x + {c_poly[2]:.12f}")
print(f"    誤差: {err_poly:.6f}")

# (b) 指數
ln_y = np.log(y)
A_exp = np.vstack([x, np.ones(len(x))]).T
c_exp, _, _, _ = np.linalg.lstsq(A_exp, ln_y, rcond=None)
a_exp = c_exp[0]
b_exp = np.exp(c_exp[1])
y_exp = b_exp * np.exp(a_exp * x)
err_exp = np.sum((y - y_exp)**2)
print(f"(b) y = {b_exp:.12f}e^({a_exp:.12f}x)")
print(f"    誤差: {err_exp:.6f}")

# (c) 冪函數
ln_x = np.log(x)
A_pow = np.vstack([ln_x, np.ones(len(x))]).T
c_pow, _, _, _ = np.linalg.lstsq(A_pow, ln_y, rcond=None)
a_pow = c_pow[0]
b_pow = np.exp(c_pow[1])
y_pow = b_pow * x**a_pow
err_pow = np.sum((y - y_pow)**2)
print(f"(c) y = {b_pow:.12f}x^{a_pow:.12f}")
print(f"    誤差: {err_pow:.6f}")

# ========== 題目2 ==========
x = sp.symbols('x')
f = sp.Rational(1, 2) * sp.cos(x) + sp.Rational(1, 4) * sp.sin(2 * x)


basis = [1, x, x ** 2]


A = sp.zeros(3)
b = sp.zeros(3, 1)
for i in range(3):
    for j in range(3):
        A[i, j] = sp.integrate(basis[i] * basis[j], (x, -1, 1))
for i in range(3):
    b[i, 0] = sp.integrate(f * basis[i], (x, -1, 1))


c = A.LUsolve(b)


coeffs = [sp.N(val, 12) for val in c]


p2 = coeffs[0] + coeffs[1] * x + coeffs[2] * x ** 2

print("Numeric coefficients (c0, c1, c2):")
print(coeffs)
print("\nLeast squares polynomial p2(x) ≈")
print(p2)


# ========== 題目3 ==========
def f(x):
    """向量化: f(x) = x^2 sin x"""
    return x**2 * np.sin(x)


m = 16               
n = 4                
xj = np.arange(m) / m  


a0 = np.sum(f(xj)) / m
ak = np.zeros(n)
bk = np.zeros(n)

for k in range(1, n + 1):
    cos_term = np.cos(2 * np.pi * k * xj)
    sin_term = np.sin(2 * np.pi * k * xj)
    ak[k - 1] = 2 * np.sum(f(xj) * cos_term) / m
    bk[k - 1] = 2 * np.sum(f(xj) * sin_term) / m


def S4_num(x):
    """Evaluate S4 using NumPy arrays or scalars."""
    res = a0
    for k in range(1, n + 1):
        res += ak[k - 1] * np.cos(2 * np.pi * k * x) + bk[k - 1] * np.sin(2 * np.pi * k * x)
    return res


xs = sp.symbols('x')
S4_sym = a0
for k in range(1, n + 1):
    S4_sym += ak[k - 1] * sp.cos(2 * sp.pi * k * xs) + bk[k - 1] * sp.sin(2 * sp.pi * k * xs)


integral_S4 = a0  


exact_integral_f = sp.integrate(xs**2 * sp.sin(xs), (xs, 0, 1))


dense_x = np.linspace(0, 1, 10001)
error_L2 = np.sqrt(np.trapezoid((f(dense_x) - S4_num(dense_x))**2, dense_x))


print("Coefficients:")
print("a0 =", a0)
for k in range(1, n + 1):
    print(f"a{k} =", ak[k - 1], ", b{k} =", bk[k - 1])

print("\nS4(x) symbolic form:")
print(sp.simplify(S4_sym))

print("\nIntegral of S4 over [0,1] =", integral_S4)
print("Integral of f over [0,1]    =", float(exact_integral_f.evalf()))
print("Continuous L2 error E(S4)    =", error_L2)


__all__ = ['a0', 'ak', 'bk', 'S4_num', 'S4_sym', 'integral_S4',
           'exact_integral_f', 'error_L2']
