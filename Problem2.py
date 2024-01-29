import numpy as np 
from method1 import *
from method2 import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
from timeit import default_timer as timer

def F(x):
    F_v = 0
    for k in range(1,n):
        if k%2 == 1 : 
            F_v += 10*(x[k-1]**2 - x[k])**2; 

        else : 
            F_v += (x[k-2]-1)**2;
    return F_v

def gradf(x):
    n = len(x)
    gradf = np.zeros_like(x)
    for k in range(1,n + 1):
        if k % 2 == 1:
            gradf[k-1] = 200*x[k-1]**3 - 200*x[k-1]*x[k] + x[k-1] - 1
        else:
            gradf[k-1] = 100*(x[k-1]- x[k-2] **2)
    return gradf


def hessf(x): 
    n = len(x)
    main_diag = np.zeros(n)
    up_sub_diag = np.zeros(n)
    for i in range(n):
        if i % 2 == 0: 
            main_diag[i] = 600 * x[i]**2 - 200 * x[i+1] + 1
            up_sub_diag[i] = -200 * x[i]
        else:  
            main_diag[i] = 100

    # Create the sparse matrix
    Hessf = sp.diags([up_sub_diag, main_diag], [-1, 0], shape=(n, n))
    Hessf = Hessf + Hessf.T - sp.diags(main_diag, 0)
    return Hessf




def report_results(f, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, fd,p, save_plots = False):
    print("========== Test Results Modifed newton method on Problem2 ==========")
    start = timer()
    k1, x_seq1, f_vals1, grad_norm_seq1 ,btseq1 = modified_newton(f, gradf, hessf, x0, kmax,tolgrad, rho, c1, btmax)
    end = timer()
    print("Time for problem2 using MN is " , end - start, "second")
    print("Minimum function value = ", f_vals1[-1])
    print("Number of iterations = ", k1)
    print("Final gradient : " , grad_norm_seq1[-1])
    print("========== Test Results Modifed newton method with FD on Problem2 ==========")
    start = timer()
    k2, x_seq2, f_vals2, grad_norm_seq2 ,btseq2 = modified_newton_FD(f,x0,kmax,fd,tolgrad,rho,c1,btmax)
    end = timer()
    print("Minimum function value = ", f_vals2[-1])
    print("Number of iterations = ", k2)
    print("Final gradient : " , grad_norm_seq2[-1])
    print("Time for Problem2 using MN and FD is " , end - start, "second")

    # Plotting for Objective Function Value
    plt.semilogy(range(k1 + 1), f_vals1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), f_vals2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Objective Function Value over iterations')
    plt.ylabel('Objective Function Value')
    plt.legend()

    if save_plots:
        plt.savefig(f'objective_function_plotPb2 for 10^{p}.png')
    plt.show()


    plt.semilogy(range(k1 + 1), grad_norm_seq1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), grad_norm_seq2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Gradient Norm  over iterations')
    plt.ylabel('Gradient Norm')
    plt.legend()

    if save_plots:
        plt.savefig(f'gradient_norm_plotPb2 for 10^{p}.png')

    plt.show()




if __name__ == '__main__':
    np.random.seed(42)  
    kmax = 1000
    dict_val = {0 : -1.2, 1: 1}
    tolgrad = 1e-6
    btmax = 100
    rho = 0.8 ## 0.3
    c1 = 1e-4
    p = 3
    n = 10**p
    x0 = np.array([dict_val[i%2] for i in range(n)])
    x_star = np.ones_like(x0)
    report_results(F, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, "C",p, save_plots = True)
