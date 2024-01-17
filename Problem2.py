## Problem 2  : (25) Extended rosenbrock function
import numpy as np 
from method1 import *
from method2 import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
def fk(x, k):
    if k % 2 == 1:
        return 10 * (x[k-1]**2 - x[k])
    else:
        return x[k-2] - 1

def F(x):
    k_values = np.arange(1, len(x) + 1)
    terms = np.vectorize(lambda k: fk(x, k))(k_values)
    return 0.5 * np.sum(terms**2)


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



def run_test(f, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho):
    print("========== Test Results Modifed newton ==========")
    # print("Starting point:", x0)
    from timeit import default_timer as timer
    start = timer()
    k, x_seq, f_vals, grad_norm_seq ,btseq = modified_newton(f, gradf, hessf, x0, kmax,tolgrad, rho, c1, btmax)
    end = timer()
    print(end - start)
    #("Final solution xk =", x_seq[-1])
    print("Minimum function value =", f_vals[-1])
    print("Number of iterations =", k)
    print(grad_norm_seq[-1])
    plt.semilogy(range(k), f_vals)
    plt.show()
    plt.semilogy(range(k), grad_norm_seq)
    plt.show()
    plt.plot(range(k)[1:], btseq)
    plt.show()


def run_test_fd(f, x0, kmax, fd, tolgrad):
    print("========== Test Results ==========")
    # print("Starting point:", x0)
    from timeit import default_timer as timer
    start = timer()
    k, x_seq, f_vals, grad_norm_seq ,btseq = modified_newton_FD(f,x0,kmax,fd,tolgrad,rho,c1,btmax)  
    end = timer()
    print("time" , end - start)
    # print(grad_norm_seq)
    #print("Final solution xk =", x_seq[-1])
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
    dict_val = {1 : -1.2, 0: 1}
    tolgrad = 1e-6
    btmax = 100
    rho = 0.4 ## 0.3
    c1 = 1e-4
    n = 10**2

    x0 = np.array([dict_val[i%2] for i in range(n)])
    ## Method 2
    run_test_fd(F,x0,kmax,"C",tolgrad)
    ## Method 1
    #run_test(F, gradf, hessf, x0, kmax, tolgrad,btmax, c1, rho)