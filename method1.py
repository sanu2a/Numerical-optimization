#Method1 : Modified Newton Method
from scipy.optimize import line_search
from scipy.sparse.linalg import gmres, lgmres, norm
import numpy as np
from tools import *
#  rho, btmax, ro, c1, parameters of armijo condition to add in tools.py and to be imported
def modified_newton(f, gradf, Hessf, x0, kmax, tolgrad=1e-5,rho = 0.5 ,c1 = 1e-4, c2 = 0.7, btmax = 10,condition_type = 'wolfe'):
    
    """
    Modified Newton method for unconstrained optimization.

    Args:
        f : The objective function to minimize.
        gradf : The gradient function of the objective function.
        Hessf : The Hessian function of the objective function.
        x0 : Starting point/initial guess.
        kmax : Maximum number of iterations.
        tolgrad : Tolerance of the gradient of f to use as a stopping criterion.
        rho
        c1
        c2
        condition_type : two options for backtracking either armijo or wolfe

    Returns:
        xk : Final solution.
        f_val : Minimum function value reached at xk.
        k : Number of iterations performed.
        x_seq : Vector containing the points generated at each iteration.
        f_vals : Vector containing the value of f at each generated point at each iteration.
    """
    
    x_seq, f_vals, btseq, grad_norm_seq = [], [], [], []
    k = 1
    xk = x0
    x_seq.append(xk)
    f_vals.append(f(xk))
    n = np.shape(x0)[0]
    gradfk = gradf(xk)
    gradfk_norm = np.linalg.norm(gradfk)
    grad_norm_seq.append(gradfk_norm)
    while k < kmax and gradfk_norm >= tolgrad:
        Hk = Hessf(xk)
        beta = 1e-3
        aii = min(Hk.diagonal())
        if aii>0:
            tau = 0
        else:
            tau = - aii + beta 
        done = False
        while not done:
            Bk = Hk + tau * np.eye(n)
            try:
                # print(1,2)
                L = np.linalg.cholesky(Bk)
                done = True
            except:
                #print("Cholesky failed")
                tau = max(10 * tau, beta)
                
        y = np.linalg.solve(L, -gradfk)
        pk = np.linalg.solve(L.T, y) 
        # y, ex1 = gmres(L, -gradfk, tol = 1e-7)
        # # if ex1 != 0 :
        # #     y = np.linalg.solve(L, -gradfk)
        # pk, ex2 = gmres(L.T, y, tol = 1e-7) 
        # if ex2 != 0 :
        #     pk = np.linalg.solve(L.T, y) 
        # pk = np.linalg.solve(Bk, -gradf)
        # pk, ex1 = gmres(Bk, -gradfk, tol = 1e-10)
        backtrackiter = 0
        alpha = 1
        while backtrackiter <  btmax:
            if armijo_condition(f,xk,alpha,c1,gradfk,pk):
                break
            alpha *= rho
            backtrackiter += 1
        if backtrackiter == btmax : 
            print("Backtracking not satisfied")
        btseq.append(backtrackiter)
        # alpha = line_search(f, gradf, xk, pk, maxiter = 100)[0]
        xk = xk + alpha * pk
        gradfk = gradf(xk)
        gradfk_norm = np.linalg.norm(gradfk)
        grad_norm_seq.append(gradfk_norm)
        x_seq.append(xk)
        f_vals.append(f(xk))
        k += 1

    if k == kmax:
        print("The optimization stopped because we reached the maximum number of iterations")

    return k, x_seq, f_vals, grad_norm_seq, btseq
