### In this file we can implement all functions that we will use in the two methods 
## including armijo conditions and finite differences for the gradient adm the hessian functions 

import numpy as np

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
    n = np.shape(x)[0]
    hessf = np.empty(shape=(n, n))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        hessf[i, i] = (f(x + h * ei) - 2 * f(x) + f(x - h * ei)) / (h**2)

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = 1
            hessf[j, i] = (f(x + h * ei + h * ej) - f(x + h * ei) - f(x + h * ej) + f(x)) / h**2
            # Assuming always working on the symmetric case
            hessf[i, j] = hessf[j, i]

    return hessf

def grad_FWFD(f, x, h):
    """
    Compute the gradient of a function using Forward Finite Differences.

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
        gradf[i] = (f(x + h * ei) - f(x)) / h

    return gradf


def armijo_condtition(f,x,alpha,c1,gradfk,pk):
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