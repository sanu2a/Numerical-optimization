### In this file we can implement all functions that we will use in the two methods 
## including armijo conditions and finite differences for the gradient adm the hessian functions 

import numpy as np
from scipy.optimize import approx_fprime

def rate(x_seq, x_star):
    e = [np.linalg.norm(x - x_star) for x in x_seq]
    q = [np.log(e[n+1]/e[n])/np.log(e[n]/e[n-1]) for n in range(1, len(e) -1 , 1)]
    return q

def grad_CFD(f, x, h):
    """
    Compute the gradient of a function using Central Finite Differences.

    Parameters:
    - f: The function to compute the gradient for.
    - x: The point at which to compute the gradient.
    - h: The step size for finite differences.

    Returns:
    - gradf: The computed gradient.
    """
    n = np.shape(x)[0]
    gradf = np.empty(shape=n)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        gradf[i] = (f(x + h * ei) - f(x - h * ei)) / (2 * h)
    return gradf



# def hessf(gradf, x, h):
#     n = np.shape(x)[0]

def hessf_FD(f, x, h):
    """
    Compute the Hessian matrix of a function using Central Finite Differences.

    Parameters:
    - f: The function to compute the Hessian for.
    - x: The point at which to compute the Hessian.
    - h: The step size for finite differences.

    Returns:
    - hessf: The computed Hessian matrix.
    """
    n = x.shape[0]
    hessf = np.zeros((n, n))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        x_plus_h_ei = x + h * ei
        x_minus_h_ei = x - h * ei

        hessf[i, i] = (f(x_plus_h_ei) - 2 * f(x) + f(x_minus_h_ei)) / (h**2)

        if i < n - 1:
            ej = np.zeros(n)
            ej[i + 1] = 1
            x_plus_h_ej = x + h * ej
            x_plus_h_ei_plus_h_ej = x + h * ei + h * ej

            hessf[i + 1, i] = (f(x_plus_h_ei_plus_h_ej) - f(x_plus_h_ei) - f(x_plus_h_ej) + f(x)) / h**2
            hessf[i, i + 1] = hessf[i + 1, i]

    return hessf



def armijo_condition(f,x,alpha,c1,gradfk,pk):
    left = f(x + alpha * pk)
    right = f(x) + c1 * alpha * np.dot(gradfk,pk)
    return left <= right

def wolfe_condition(f, x, alpha, c1, c2, grad, pk):
    armijo = armijo_condition(f, x, alpha, c1, grad, pk)
    
    if not armijo:
        return False
    
    # Curvature condition (Strong Wolfe condition)
    slope_condition = np.dot(grad, pk)
    left = np.dot(grad, (x + alpha * pk - x))
    right = c2 * slope_condition
    return left >= right


condition_function = {'armijo':armijo_condition,'wolfe':wolfe_condition}
def backtrack_condition(condition_type,f,x,alpha,c1,c2,gradfk,pk):
    condition_func = condition_function.get(condition_type)
    if condition_func == armijo_condition:
        return condition_func(f,x,alpha,c1,gradfk,pk)
    elif condition_func == wolfe_condition:
        return condition_func(f,x,alpha,c1,c2,gradfk,pk)
    else:
        raise ValueError("Invalid condition type")