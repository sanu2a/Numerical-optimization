#Method2 : modified newton method with finite differences 

import numpy as np
from tools import *
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla



#  rho, btmax, c1, parameters of armijo condition to add in tools.py and to be imported
def modified_newton_FD(f, x0, kmax, fd, tolgrad, rho ,c1 , btmax, precond = False):
    """
    Modified Newton method with finite differences.

    Args:
        f : The objective function to minimize.
        x0 : Starting point/initial guess.
        kmax : Maximum number of iterations.
        fd : Finite differences method ('c' for central, 'f' for forward, 'None' for default).
        tolgrad : Tolerance of the gradient of f to use as a stopping criterion.

    Returns:
        xk : Final solution.
        f_val : Minimum function value reached at xk.
        k : Number of iterations performed.
        x_seq : Vector containing the points generated at each iteration.
        f_vals : Vector containing the value of f at each generated point at each iteration.
    """
    
    hgrad = np.sqrt(np.finfo(float).eps)/2
    hhess = np.sqrt(hgrad)
    gradf_fd = grad_CFD
    hessf_fd = hessf_FD
    n = x0.shape[0]

    x_seq, f_vals = [], []
    k = 0
    xk = x0
    x_seq.append(xk)
    f_vals.append(f(xk))
    n = np.shape(x0)[0]
    grad_norm_seq , btseq = [], []
    gradfk = gradf_fd(f, xk, hgrad)
    gradfk_norm = np.linalg.norm(gradfk)
    grad_norm_seq.append(gradfk_norm)
    beta = 1e-3

    while k < kmax and gradfk_norm >= tolgrad:
        Hk = hessf_fd(f, xk, hhess)
        aii = min(np.diag(Hk))
        if aii > 0:
            tau = 0
        else:
            tau =  - aii + beta 
            
        done = False
        while not done:
            Bk = Hk + tau * np.eye(n)
            try:
                L = np.linalg.cholesky(Bk)
                done = True
            except:
                tau = max(10 * tau, beta)
        if precond == False:
            y = np.linalg.solve(L, -gradfk)
            pk = np.linalg.solve(L.T, y) 
        else : 
            M2 = spla.spilu(Bk)
            M_x = lambda x: M2.solve(x)
            M = spla.LinearOperator((n,n), M_x)
            pk, _ = spla.gmres(Bk,-gradfk,M=M)
            
        alpha = 1
        backtrackiter = 0
        while backtrackiter <  btmax:
            if armijo_condition(f,xk,alpha,c1,gradfk,pk):
                break
            alpha *= rho
            backtrackiter += 1

        btseq.append(backtrackiter)
        xk = xk + alpha * pk
        x_seq.append(xk)
        f_vals.append(f(xk))
        gradfk = gradf_fd(f, xk, hgrad)
        gradfk_norm = np.linalg.norm(gradfk)
        grad_norm_seq.append(gradfk_norm)
        k += 1

    if k == kmax:
        print("The optimization stopped because we reached the maximum number of iterations")

    return k, x_seq, f_vals, grad_norm_seq, btseq
