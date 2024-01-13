## Problems : 25 

## Problem 2 (25) : Extended 

from math import exp
import numpy as np 
from method1 import *
from method2 import *


def F(x): 
    n = np.shape(x)[0]
    Fx = 0
    for i in range(1, n):
        Fx += (x[i-1]**2)**(x[i]**2 + 1) + (x[i]**2)**(x[i-1]**2 + 1)
    return Fx


def F1(x):
    n = np.shape(x)[0]
    Fx = 0
    k = n/2
    for j in range(1, k):
        i = 2*j
        Fx += (x[i-1]**2)**(x[i]**2 + 1) + (x[i]**2)**(x[i-1]**2 + 1)
    return Fx 

def gradf(x):
    n = np.shape(x)[0]
    dx = sp.symbols(' '.join([f'x{i}' for i in range(n)]))
    grad_F = [sp.diff(F(dx), var) for var in dx]
    grad_F_func = sp.lambdify(dx, grad_F, 'numpy')
    return np.array(grad_F_func(*x))

def hessf(x):
    n = len(x)
    dx = sp.symbols(' '.join([f'x{i}' for i in range(n)]))
    hessian_F = sp.hessian(F(dx), dx)
    hessian_F_func = sp.lambdify(dx, hessian_F, 'numpy')
    return hessian_F_func(*x)
    
def run_test_fd(method, f, x0, kmax, fd, tolgrad):
    dictio = {"c" : "Centred", "FW" : "Forward"} 
    print(f"========== Test Results {dictio[fd]} ==========")
    print("Method:", method.__name__)
    xk, fk, k, x_seq, f_vals = method(f, x0, kmax, fd, tolgrad)
    print("Final solution xk =", xk)
    print("MSE of the solution !", np.linalg.norm(xk))
    print("Minimum function value =", round(fk, 3))
    print("Number of iterations =", k)

def run_test(method, f, gradf, hessf, x0, kmax, tolgrad=1e-5):
    print("========== Test Results ==========")
    print("Method:", method.__name__)
    print("Starting point:", x0)
    xk, fk, k, x_seq, f_vals = method(f, gradf, hessf, x0, kmax, tolgrad)
    print("Final solution xk =", xk)
    print("MSE of the solution !", np.linalg.norm(xk))
    print("Minimum function value =", round(fk, 3))
    print("Number of iterations =", k)
    
if __name__ == '__main__':
    np.random.seed(42)  
    kmax = 1000
    tolgrad = 1e-7
    dict_val = {1: -1, 0: 1}

    n = 10 * 3
    x0 = np.array([dict_val[i % 2] for i in range(n)])
    print(F1(x0))
    print(F(x0))
    
    # # Test modified_newton method
    # run_test(modified_newton, F, gradf, hessf, x0, kmax, tolgrad)

    # # Test modified_newton_FD method with Forward Finite Differences
    # run_test_fd(modified_newton_FD, F, x0, kmax, "FW", tolgrad)

    # # Test modified_newton_FD method with Centred Finite Differences
    # run_test_fd(modified_newton_FD, F, x0, kmax, "c", tolgrad)
