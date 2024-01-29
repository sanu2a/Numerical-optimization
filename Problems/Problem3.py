import numpy as np
from method1 import *
from method2 import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import numpy as np
from scipy.sparse import diags


def F(x):
    n = len(x)
    F = 0
    f_1 = x[0] - 1
    F += f_1**2
    for i in range(1, n):
        F += (10 * i * (x[i] - x[i-1])**2)**2
    value = F / 2
    return value


def gradf(x):
    n = len(x)
    gradient = np.zeros(n)
    gradient[0] = x[0] - 1 + 200*(x[0]-x[1])**3
    for k in range(1, n-1):
        gradient[k] = 200*(k**2)*(x[k]-x[k+1])**3 - 200*((k-1)**2)*(x[k-1]-x[k])**3
    gradient[n-1] = -200*((n-1)**2)*((x[n-2]-x[n-1])**3)
    return gradient


def hessf(x):
    n = len(x)
    up_sub_diagonal = np.zeros(n)
    main_diagonal = np.zeros(n)
    up_sub_diagonal[0] = -50*12*((x[1]-x[0])**2)
    main_diagonal[0] = 1 + 50*12*((x[1]-x[0])**2)
    for k in range(1, n-1):
        up_sub_diagonal[k] = -50*12*((k+1)**2)*(x[k]-x[k+1])**2
        main_diagonal[k] = 12*50*((k)**2)*((x[k-1]-x[k])**2) + 12*50*((k+1)**2)*((x[k]-x[k+1])**2)
    main_diagonal[n-1] = 50*12*((n-1)**2)*(x[n-2]-x[n-1])**2
    Hessian = diags([up_sub_diagonal, main_diagonal], [-1, 0], shape=(n, n)) + \
              diags([up_sub_diagonal], [-1], shape=(n, n)).transpose()
    return Hessian




def rate(x_seq):
    differences = [np.linalg.norm(x_seq[n] - x_seq[n - 1]) for n in range(1, len(x_seq))]
    q = [np.log(differences[n] / differences[n - 1]) / np.log(differences[n - 1] / differences[n - 2]) for n in range(2, len(differences))]
    return q

def report_results(f, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, fd,p, save_plots = False, precond = False):
    print("========== Test Results Modifed newton method on Problem2 ==========")
    start = timer()
    k1, x_seq1, f_vals1, grad_norm_seq1 ,btseq1 = modified_newton(f, gradf, hessf, x0, kmax,tolgrad, rho, c1, btmax, precond = precond)
    end = timer()
    print("Time for problem1 using MN is " , end - start, "second")
    print("Minimum function value = ", f_vals1[-1])
    print("Number of iterations = ", k1)
    print("Final gradient : " , grad_norm_seq1[-1])
    print("========== Test Results Modifed newton method with FD on Problem2 ==========")
    start = timer()
    k2, x_seq2, f_vals2, grad_norm_seq2 ,btseq2 = modified_newton_FD(f,x0,kmax,fd,tolgrad,rho,c1,btmax, precond = False)
    end = timer()
    print("Minimum function value = ", f_vals2[-1])
    print("Number of iterations = ", k2)
    print("Final gradient : " , grad_norm_seq2[-1])
    print("Time for Problem1 using MN and FD is " , end - start, "second")

    # Plotting for Objective Function Value
    plt.semilogy(range(k1 + 1), f_vals1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), f_vals2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Objective Function Value over iterations')
    plt.ylabel('Objective Function Value')
    plt.legend()

    if save_plots:
        plt.savefig(f'objective_function_plotPb3{precond} 10^{p}.png')
    plt.show()


    plt.semilogy(range(k1 + 1), grad_norm_seq1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), grad_norm_seq2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Gradient Norm  over iterations')
    plt.ylabel('Gradient Norm')
    plt.legend()

    if save_plots:
        plt.savefig(f'gradient_norm_plotPb3{precond}10^{p}.png')

    plt.show()
    rate1 = rate(x_seq1[1:])
    rate2 = rate(x_seq2[1:])
    mean_rate1 = np.mean(rate1)
    mean_rate2 = np.mean(rate2)

    # Plot the convergence rate for each method
    plt.plot(range(k1-3), rate1, label='rate_cv_MN')
    plt.plot(range(k2-3), rate2, label='rate_cv_MNFD')

    # Add vertical lines for the mean convergence rate
    plt.axhline(mean_rate1, color='red', linestyle='--', label=f'Mean Rate MN: {mean_rate1:.2f}')
    plt.axhline(mean_rate2, color='blue', linestyle='--', label=f'Mean Rate MNFD: {mean_rate2:.2f}')

    plt.xlabel('Iterations')
    plt.title(f'Convergence Rate Of Implemented Methods')
    plt.ylabel('Convergence rate')
    plt.legend()
    plt.savefig(f'cv_rate_Pb3_for_10^{p}.png')
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)  
    kmax = 2000 
    tolgrad = 1e-5
    btmax = 100
    rho =  0.4 ## 0.3
    c1 = 1e-4
    p = 3
    n = 10**p

    # Starting point
    x0 = np.array([-1.2 for i in range(n-1)])
    x0 = np.append(x0, -1)
    x_star = np.ones_like(x0)
    #report_results(F, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, "C",p, save_plots = True, precond = True)
    report_results(F, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, "C", p,save_plots = True, precond = False)

