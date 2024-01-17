## Problem 75 
import numpy as np 
from method1 import *
from method2 import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
def fk(x, k):
    if k != 1:
        return 10*(k-1)*(x[k-1] - x[k-2])**2
    else:
        return x[k-2]

def F(x):
    k_values = np.arange(1, len(x) + 1)
    terms = np.vectorize(lambda k: fk(x, k))(k_values)
    return 0.5 * np.sum(terms**2)


def gradf(x):
    ##  TO BE IMPLEMENTED
    return 


def hessf(x):
    ## TO BE IMPLEMENTED
    return 



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
    tolgrad = 1e-6
    btmax = 100
    rho = 0.4 ## 0.3
    c1 = 1e-4
    n = 10**3
    # Starting point
    x0 = np.array([-1.2 for i in range(n-1)])
    x0 = np.append(x0, 1)
    
    ## Method 1
    run_test_fd(F,x0,kmax,"C",tolgrad)
    ## Method 2 
    #run_test(F, gradf, hessf, x0, kmax, tolgrad,btmax, c1, rho)