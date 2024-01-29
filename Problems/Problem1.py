## Problem 1 (1) : Chained Roseenbrock Function
import numpy as np 
from method1 import *
from method2 import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
from timeit import default_timer as timer

def F(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def gradf(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def hessf(x):
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H




def compute_rate_of_convergence(x_seq, x_star):
    # Compute the rate of convergence using the formula
    # Rate of Convergence = log(error_{k+1} / error_k) / log(step_{k+1} / step_k)
    errors = [norm(x - x_star) for x in x_seq]
    steps = [norm(x_seq[i + 1] - x_seq[i]) for i in range(len(x_seq) - 1)]

    log_ratios = [np.log(errors[i + 1] / errors[i]) / np.log(steps[i + 1] / steps[i]) for i in range(len(errors) - 1)]
    
    return np.mean(log_ratios)

def report_results(f, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, fd,p, save_plots = False):
    print("========== Test Results Modifed newton method with FD on Problem1 ==========")
    start = timer()
    k2, x_seq2, f_vals2, grad_norm_seq2 ,btseq2 = modified_newton_FD(f,x0,kmax,fd,tolgrad,rho,c1,btmax)
    end = timer()
    print("Minimum function value = ", f_vals2[-1])
    print("Number of iterations = ", k2)
    print("Final gradient : " , grad_norm_seq2[-1])
    print("Time for Problem1 using MN and FD is " , end - start, "second")

    print("========== Test Results Modifed newton method on Problem1 ==========")
    start = timer()
    k1, x_seq1, f_vals1, grad_norm_seq1 ,btseq1 = modified_newton(f, gradf, hessf, x0, kmax,tolgrad, rho, c1, btmax)
    end = timer()
    print("Time for problem1 using MN is " , end - start, "second")
    print("Minimum function value = ", f_vals1[-1])
    print("Number of iterations = ", k1)
    print("Final gradient : " , grad_norm_seq1[-1])
    # Plotting for Objective Function Value
    plt.semilogy(range(k1 + 1), f_vals1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), f_vals2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Objective Function Value over iterations')
    plt.ylabel('Objective Function Value')
    plt.legend()

    if save_plots:
        plt.savefig(f'objective_function_plotPb1 for 10^{p}.png')
    plt.show()


    plt.semilogy(range(k1 + 1), grad_norm_seq1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), grad_norm_seq2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Gradient Norm  over iterations')
    plt.ylabel('Gradient Norm')
    plt.legend()

    if save_plots:
        plt.savefig(f'gradient_norm_plotPb1 for 10^{p}.png')

    plt.show()
    rate1 = rate(x_seq1[1:], x_star)
    rate2 = rate(x_seq2[1:], x_star)
    mean_rate1 = np.mean(rate1)
    mean_rate2 = np.mean(rate2)

    # Plot the convergence rate for each method
    plt.plot(range(k1-2), rate1, label='rate_cv_MN')
    plt.plot(range(k2-2), rate2, label='rate_cv_MNFD')

    # Add vertical lines for the mean convergence rate
    plt.axhline(mean_rate1, color='red', linestyle='--', label=f'Mean Rate MN: {mean_rate1:.2f}')
    plt.axhline(mean_rate2, color='blue', linestyle='--', label=f'Mean Rate MNFD: {mean_rate2:.2f}')

    plt.xlabel('Iterations')
    plt.title(f'Convergence Rate Of Implemented Methods')
    plt.ylabel('Convergence rate')
    plt.legend()
    plt.savefig(f'cv_rate_Pb1_for_10^{p}.png')
    plt.show()

    
    


if __name__ == '__main__':
    np.random.seed(42)  
    kmax = 1600 
    dict_val = {1 : 1, 0: -1.2}
    tolgrad = 1e-6
    btmax = 100
    rho = 0.4 ## 0.3
    c1 = 1e-4
    p = 3
    n = 10**p
    x0 = np.array([dict_val[i%2] for i in range(n)])
    x_star = np.ones_like(x0)
    report_results(F, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, "C",p, save_plots = True)

    