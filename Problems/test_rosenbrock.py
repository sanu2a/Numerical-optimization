from method1 import modified_newton
from method2 import modified_newton_FD
import numpy as np
import numpy as np
from method1 import modified_newton
from method2 import modified_newton_FD
from timeit import default_timer as timer
import matplotlib.pyplot as plt 
from tools import *



def Rosenbrock():
    """
    Return handle functions representing the Rosenbrock function.

    Returns:
        f: Handle function for the Rosenbrock function.
        gradf: Handle function for the gradient of the Rosenbrock function.
        hessf: Handle function for the Hessian of the Rosenbrock function.
    """
    f = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    dfx1 = lambda x: 2 * (200 * x[0]**3 - 200 * x[0] * x[1] + x[0] - 1)
    dfx2 = lambda x: 200 * (x[1] - x[0]**2)
    gradf = lambda x: np.array([dfx1(x), dfx2(x)])
    h11 = lambda x: 400 * (3 * x[0]**2 - x[1]) + 2
    h12 = lambda x: -400 * x[0]
    h21 = lambda x: -400 * x[0]
    h22 = lambda x: 200
    hessf = lambda x: np.array([[h11(x), h12(x)], [h21(x), h22(x)]])
    return f, gradf, hessf



def report_results(f, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho, x_star, fd, save_plots = True):
    print("========== Test Results Modifed newton method on rosenbrock function ==========")
    print("Starting point:", x0)
    start = timer()
    k1, x_seq1, f_vals1, grad_norm_seq1 ,btseq1 = modified_newton(f, gradf, hessf, x0, kmax,tolgrad, rho, c1, btmax)
    end = timer()
    print("Time for Rosenbrock in R^2 using MN is " , end - start, "second")
    print("Minimum function value = ", f_vals1[-1])
    print("Number of iterations = ", k1)
    print("Final gradient : " , grad_norm_seq1[-1])
    print("========== Test Results Modifed newton method with FD on rosenbrock function ==========")
    print("Starting point:", x0)
    start = timer()
    k2, x_seq2, f_vals2, grad_norm_seq2 ,btseq2 = modified_newton_FD(f,x0,kmax,fd,tolgrad,rho,c1,btmax)
    end = timer()
    print("Minimum function value = ", f_vals2[-1])
    print("Number of iterations = ", k2)
    print("Final gradient : " , grad_norm_seq2[-1])
    print("norm ", np.linalg.norm(x0 - x_star))
    print("Time for Rosenbrock in R^2 using MN and FD is " , end - start, "second")

    # Plotting for Objective Function Value
    plt.semilogy(range(k1 + 1), f_vals1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), f_vals2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Objective Function Value over iterations\nStarting point: {x0}')
    plt.ylabel('Objective Function Value')
    plt.legend()

    if save_plots:
        plt.savefig(f'objective_function_plot{x0}.png')
    plt.show()

    # Initialize figure 

    # Plotting for Gradient Norm
    plt.semilogy(range(k1 + 1), grad_norm_seq1, label='MN with Exact Hessian')
    plt.semilogy(range(k2 + 1), grad_norm_seq2, label='MN with CFD')
    plt.xlabel('Iterations')
    plt.title(f'Gradient Norm over iterations\nStarting point: {x0}')
    plt.ylabel('Gradient Norm')
    plt.legend()

    if save_plots:
        plt.savefig(f'gradient_norm_plot{x0}.png')

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
    plt.savefig(f'cv_rate_rosen R2{x0}.png')
    plt.show()





if __name__ == '__main__':
    np.random.seed(42)  
    f, gradf, hessf = Rosenbrock()
    x01 = np.array([1.2, 1.2])
    x02 = np.array([-1.2, 1])
    kmax = 1000
    tolgrad = 1e-8
    btmax = 50
    rho = 0.4
    c1 = 1e-4
    x_star = np.array([1,1])
    report_results(f, gradf, hessf, x01, kmax, tolgrad, btmax, c1, rho, x_star, "C", True)
    report_results(f, gradf, hessf, x02, kmax, tolgrad, btmax, c1, rho, x_star, "C", True)


    
    
          
    