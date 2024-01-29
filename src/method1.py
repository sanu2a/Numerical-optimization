#Method1 : Modified Newton Method
from scipy.optimize import line_search
from scipy.sparse.linalg import gmres
import numpy as np
from tools import *
import scipy.sparse.linalg as spla

#  rho, btmax, ro, c1, parameters of armijo condition to add in tools.py and to be imported
import numpy as np
from scipy.sparse.linalg import spilu, LinearOperator
from scipy.sparse import csc_matrix


def modified_newton(f, gradf, Hessf, x0, kmax, tolgrad=1e-5,rho = 0.5 ,c1 = 1e-4, c2 = 0.9, btmax = 10,condition_type = 'wolfe',stop = 50, precond = False):
    
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
    # if precond == True :
    #     print("Preconditioning to be used")
    x_seq, f_vals, btseq, grad_norm_seq = [], [], [], []
    btseq = []
    k = 0
    xk = x0
    x_seq.append(xk)
    fk = f(xk)
    f_vals.append(fk)
    n = np.shape(x0)[0]
    gradfk = gradf(xk)
    gradfk_norm = np.linalg.norm(gradfk)
    grad_norm_seq.append(gradfk_norm)
    beta = 1e-3
    while k < kmax and gradfk_norm >= tolgrad:
        Hk = Hessf(xk)
        aii = min(Hk.diagonal())
        if aii>0:
            tau = 0
        else:
            tau = - aii + beta 
        done = False
        j = 0
        while not done and j<stop:
            Bk = Hk + tau * np.eye(n)
            try:
                L = np.linalg.cholesky(Bk)
                done = True
            except:
                #print("Cholesky failed")
                tau = max(10 * tau, beta)
            j+=1
            

        if precond == False :  
            y = np.linalg.solve(L, -gradfk)
            pk = np.linalg.solve(L.T, y) 
        else : 
            try :
                M2 = spla.spilu(Bk)
                M_x = lambda x: M2.solve(x)
                M = spla.LinearOperator((n,n), M_x)
                pk, _ = spla.gmres(Bk,-gradfk,M=M)
            except :
                y = np.linalg.solve(L, -gradfk)
                pk = np.linalg.solve(L.T, y) 
        alpha = 1
        backtrackiter = 0
        while backtrackiter <  btmax:
            if armijo_condition(f,xk,alpha,c1,gradfk,pk):
                break
            alpha *= rho
            backtrackiter += 1
        
        btseq.append(backtrackiter)
        #alpha = line_search(f, gradf, xk, pk, maxiter = 100, c1=c1, c2 = 0.9, gfk=gradfk, old_fval= fk, old_old_fval = old_old_fval)[0]
        xk = xk + alpha * pk
        gradfk = gradf(xk)
        gradfk_norm = np.linalg.norm(gradfk)
        grad_norm_seq.append(gradfk_norm)
        x_seq.append(xk)
        fk = f(xk)
        f_vals.append(fk)
        k += 1

    if k == kmax:
        print("The optimization stopped because we reached the maximum number of iterations")

    return k, x_seq, f_vals, grad_norm_seq, btseq
