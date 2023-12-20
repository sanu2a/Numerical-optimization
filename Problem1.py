## Problems : 73 / 12 / 56 ? 

## Problem 1 (12) : Generalization of the Brown function 1
## Problem 1 (13) : Generalization of the Brown function 2

from math import exp
import numpy as np 
from method1 import *
from method2 import *


def F(x): 
    
    n = np.shape(x)[0]
    Fx = 0
    ## Generalization 1 ?
    # for i in range(2, n):
    #     Fx += (x[i-1] - 3)**2  + (x[i - 1] - x[i]) + exp(20*x[i-1] - x[i])

    ## Generalization 2
    for i in range(1, n):
        Fx += (x[i-1]**2)**(x[i]**2 + 1) + (x[i]**2)**(x[i-1]**2 + 1)
 
    return Fx

# def gradf(x): 
#     n = np.shape(x)[0]
#     gradF = np.zeros(n)
    
#     for i in range(2, n):
#         gradF[i-2] += 2 * (x[i-2]**2)**(x[i-1]**2 + 1) * x[i-1]
#         gradF[i-1] += 2 * (x[i-1]**2)**(x[i-2]**2 + 1) * x[i-2]
    
#     return gradF

# def run_test(method, f, gradf, hessf, x0, kmax, tolgrad=1e-5):
#     print("========== Test Results ==========")
#     print("Method:", method.__name__)
#     print("Starting point:", x0)
#     xk, fk, k, x_seq, f_vals = method(f, gradf, hessf, x0, kmax, tolgrad)
#     print("Final solution xk =", xk)
#     print("Minimum function value =", round(fk, 3))
#     print("Number of iterations =", k)


def run_test_fd(method, f, x0, kmax, fd, tolgrad):
    dictio = {"c" : "Centred", "FW" : "Forward"} 
    print(f"========== Test Results {dictio[fd]} ==========")
    print("Method:", method.__name__)
    #print("Starting point:", x0)
    xk, fk, k, x_seq, f_vals = method(f, x0, kmax, fd, tolgrad)
    #print("Final solution xk =", xk)
    print("MSE of the solution !", np.linalg.norm(xk))
    print("Minimum function value =", round(fk, 3))
    print("Number of iterations =", k)
    
if __name__ == '__main__':
    np.random.seed(42)  
    kmax = 1000
    tolgrad = 1e-7
    dict_val = {1 : -1, 0: 1}

    n = 10*3
    #x0 = np.array([dict_val[i%2] for i in range(n)])
    x0 = np.array([dict_val[i%2] for i in range(n)])
    ## MNM with FD ! 
    run_test_fd(modified_newton_FD, F, x0, kmax,"FW", tolgrad)
    run_test_fd(modified_newton_FD, F, x0, kmax, "c", tolgrad)