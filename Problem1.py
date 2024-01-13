## Problem 1 (1) : Chained Roseenbrock Function
import numpy as np 
from method1 import *
from method2 import *
import matplotlib.pyplot as plt

def F(x): 
    n = np.shape(x)[0]
    Fx = 0
    for i in range(1, n):
        Fx += 100*(x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2
    return Fx

def gradf(p):
    N = len(p)
    df_dp = np.zeros(N)
    # First partial derivative
    df_dp[0] = -400 * p[0] * (p[1] - p[0]**2) - 2 * (1 - p[0])
    # Intermediate partial derivatives
    for j in range(1, N-1):
        df_dp[j] = 200 * (p[j] - p[j - 1]**2) - 400 * p[j] * (p[j+1] - p[j]**2) - 2 * (1 - p[j])
    # Last partial derivativep
    df_dp[N - 1] = 200 * (p[N - 1] - p[N - 2]**2)

    return df_dp


def hessf(x):
    n = len(x)
    hessian = np.zeros((n, n))
    # Diagonal elements
    hessian[0, 0] = 1200 * x[0]**2 - 400 * x[1] + 2
    hessian[n - 1, n - 1] = 200

    # Off-diagonal elements
    for i in range(1, n - 1):
        # Diagonal elements
        hessian[i, i] = 202 + 1200 * x[i]**2 - 400 * x[i + 1]
        # Upper diagonal elements
        hessian[i, i - 1] =  hessian[i - 1, i ] = -400 * x[i - 1]
        # # # Lower diagonal elements
        hessian[i , i + 1] =  hessian[i+1, i] =  -400 * x[i]
    return hessian




def run_test(f, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho):
    print("========== Test Results Modifed newton ==========")
    # print("Starting point:", x0)
    k, x_seq, f_vals, grad_norm_seq ,btseq = modified_newton(f, gradf, hessf, x0, kmax,tolgrad, rho, c1, btmax)
    #("Final solution xk =", x_seq[-1])
    print("Minimum function value =", f_vals[-1])
    print("Number of iterations =", k)
    print(grad_norm_seq[-1])
    plt.plot(range(k), f_vals)
    plt.show()
    plt.plot(range(k), grad_norm_seq)
    plt.show()
    plt.plot(range(k)[1:], btseq)
    plt.show()


def run_test_fd(f, x0, kmax, fd, tolgrad):
    print("========== Test Results ==========")
    # print("Starting point:", x0)
    k, x_seq, f_vals, grad_norm_seq ,btseq = modified_newton_FD(f,x0,kmax,fd,tolgrad,rho,c1,btmax)
    # print(grad_norm_seq)
    print("Final solution xk =", x_seq[-1])
    print("Minimum function value =", f_vals[-1])
    print("Number of iterations =", k)
    print(grad_norm_seq[-1])
    plt.plot(range(k), f_vals)
    plt.show()
    plt.plot(range(k), grad_norm_seq)
    plt.show()
    plt.plot(range(k)[1:], btseq)
    plt.show()




if __name__ == '__main__':
    np.random.seed(42)  
    kmax = 1000
    dict_val = {1 : -1, 0: 1}
    tolgrad = 1e-5
    btmax = 70
    rho = 0.8
    c1 = 1e-4
    n = 10**3

    #x0 = np.array([dict_val[i%2] for i in range(n)])
    x0 = np.array([dict_val[i%2] for i in range(n)])
    # run_test_fd(F,x0,kmax,"C",tolgrad)
    run_test(F, gradf, hessf, x0, kmax, tolgrad,btmax, c1, rho)