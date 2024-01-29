# In thsi file we will be impelementing the steepest descent 
import numpy as np

def steepest_desc_bcktrck(x0, f, gradf, alpha0, kmax, tolgrad, c1, rho, btmax):
    # Function handle for the Armijo condition
    def armijo_condition(fk, alpha, gradfk, pk):
        return fk + c1 * alpha * np.dot(gradfk, pk)

    # Initializations
    n = len(x0)
    xseq = np.zeros((n, kmax))
    btseq = np.zeros(kmax)

    xk = np.copy(x0)
    fk = f(xk)
    gradfk = gradf(xk)
    k = 0
    gradfk_norm = np.linalg.norm(gradfk)

    while k < kmax and gradfk_norm >= tolgrad:
        # Compute the descent direction
        pk = -gradf(xk)

        # Reset the value of alpha
        alpha = alpha0

        # Compute the candidate new xk
        xnew = xk + alpha * pk
        # Compute the value of f in the candidate new xk
        fnew = f(xnew)

        bt = 0
        # Backtracking strategy:
        # 2nd condition is the Armijo condition not satisfied
        while bt < btmax and fnew > armijo_condition(fk, alpha, gradfk, pk):
            # Reduce the value of alpha
            alpha = rho * alpha
            # Update xnew and fnew w.r.t. the reduced alpha
            xnew = xk + alpha * pk
            fnew = f(xnew)

            # Increase the counter by one
            bt += 1

        # Update xk, fk, gradfk_norm
        xk = np.copy(xnew)
        fk = fnew
        gradfk = gradf(xk)
        gradfk_norm = np.linalg.norm(gradfk)

        # Increase the step by one
        k += 1

        # Store current xk in xseq
        #xseq[:, k-1] = xk
        # Store bt iterations in btseq
        #btseq[k-1] = bt

    # "Cut" xseq and btseq to the correct size
    #xseq = xseq[:, :k]
   # btseq = btseq[:k]

    return xk, fk, gradfk_norm, k, xseq, btseq

