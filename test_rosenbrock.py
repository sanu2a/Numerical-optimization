from method1 import modified_newton
from method2 import modified_newton_FD
import numpy as np
import numpy as np
from method1 import modified_newton
from method2 import modified_newton_FD

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

def run_test(method, f, gradf, hessf, x0, kmax, tolgrad=1e-5):
    print("========== Test Results ==========")
    print("Method:", method.__name__)
    print("Starting point:", x0)
    xk, fk, k, x_seq, f_vals = method(f, gradf, hessf, x0, kmax, tolgrad)
    print("Final solution xk =", xk)
    print("Minimum function value =", round(fk, 3))
    print("Number of iterations =", k)


def run_test_fd(method, f, x0, kmax, fd, tolgrad):
    dictio = {"c" : "Centred", "FW" : "Forward"} 
    print(f"========== Test Results {dictio[fd]} ==========")
    print("Method:", method.__name__)
    print("Starting point:", x0)
    xk, fk, k, x_seq, f_vals = method(f, x0, kmax, fd, tolgrad)
    print("Final solution xk =", xk)
    print("Minimum function value =", round(fk, 3))
    print("Number of iterations =", k)
if __name__ == '__main__':
    np.random.seed(42)  
    f, gradf, hessf = Rosenbrock()
    x01 = np.array([1.2, 1.2])
    x02 = np.array([-1.2, 1])
    kmax = 1000
    tolgrad = 1e-5

    run_test(modified_newton, f, gradf, hessf, x01, kmax, tolgrad)
    run_test(modified_newton, f, gradf, hessf, x02, kmax, tolgrad)

    run_test_fd(modified_newton_FD, f, x01, kmax,"c", tolgrad)
    run_test_fd(modified_newton_FD, f, x02, kmax, "FW", tolgrad)


    
    
          
    
