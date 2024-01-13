from method1 import modified_newton
from method2 import modified_newton_FD
import numpy as np
import numpy as np
from method1 import modified_newton
from method2 import modified_newton_FD
import matplotlib.pyplot as plt 

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

def run_test(f, gradf, hessf, x0, kmax, tolgrad, btmax, c1, rho):
    print("========== Test Results Modifed newton metod on rosenbrock function ==========")
    print("Starting point:", x0)
    k, x_seq, f_vals, grad_norm_seq ,btseq = modified_newton(f, gradf, hessf, x0, kmax,tolgrad, rho, c1, btmax)
    #print("Final solution xk =", x_seq[-1])
    print("Minimum function value =", f_vals[-1])
    print("Number of iterations =", k)
    print("Finsl gradient : " , grad_norm_seq[-1])
    plt.plot(range(k), f_vals)
    plt.show()
    plt.plot(range(k), grad_norm_seq)
    plt.show()
    


def run_test_fd(f, x0, kmax, fd, tolgrad):
    print("========== Test Results ==========")
    print("Starting point:", x0)
    k, x_seq, f_vals, grad_norm_seq ,btseq = modified_newton_FD(f,x0,kmax,fd,tolgrad,rho,c1,btmax)
    # print(grad_norm_seq)
    #print("Final solution xk =", x_seq[-1])
    print("Minimum function value =", f_vals[-1])
    print("Number of iterations =", k)
    print("Finsl gradient : " , grad_norm_seq[-1])
    plt.plot(range(k), f_vals)
    plt.show()
    plt.plot(range(k), grad_norm_seq)
    plt.show()
    plt.plot(range(k)[1:], btseq)
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)  
    f, gradf, hessf = Rosenbrock()
    x01 = np.array([1.2, 1.2])
    x02 = np.array([-1.2, 1])
    kmax = 1000
    tolgrad = 1e-7
    btmax = 50
    rho = 0.8
    c1 = 1e-4
    run_test(f, gradf, hessf, x02, kmax, tolgrad,btmax, c1, rho)
    # run_test(f, gradf, hessf, x01, kmax, tolgrad,btmax, c1, rho)
    ## For the finite differences the centred is more accurate than the FW
    # run_test_fd(f,x02,kmax,"C",tolgrad)
    # run_test_fd(f,x01,kmax,"C",tolgrad)



    
    
          
    