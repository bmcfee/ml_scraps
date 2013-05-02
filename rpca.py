'''Robust PCA in python.'''
# CREATED:2013-04-29 08:50:44 by Brian McFee <brm2132@columbia.edu>
#  
import numpy as np
import scipy
import scipy.linalg
import scipy.weave

def nuclear_prox(A, r=1.0):
    '''Proximal operator for scaled nuclear norm:
    Y* <- argmin_Y  r * ||Y||_* + 1/2 * ||Y - A||_F^2

    Arguments:
        A    -- (ndarray) input matrix
        r    -- (float>0) scaling factor

    Returns:
        Y    -- (ndarray) if A = USV', then Y = UTV'
                          where T = max(S - r, 0)
    '''
    
    U, S, V = scipy.linalg.svd(A, full_matrices=False)
    T = np.maximum(S - r, 0.0)
    Y = (U * T).dot(V)
    return Y

def l1_prox(A, r=1.0):
    '''Proximal operator for entry-wise matrix l1 norm:
    Y* <- argmin_Y r * ||Y||_1 + 1/2 * ||Y - A||_F^2

    Arguments:
        Arguments:
        A    -- (ndarray) input matrix
        r    -- (float>0) scaling factor

    Returns:
        Y    -- (ndarray) Y = A after shrinkage
    '''
    
    Y = np.empty_like(A)
    
    numel = A.size
    
    shrinkage = r"""
    for (int i = 0; i < numel; i++) {
        Y[i] = 0;

        if (A[i] - r > 0) {
            Y[i] = A[i] - r;
        } else if (A[i] + r <= 0) {
            Y[i] = A[i] + r;
        }
    }
    """
    
    scipy.weave.inline(shrinkage, ['numel', 'A', 'r', 'Y'])
    return Y

def robust_pca_cost(Y, Z, alpha):
    '''Get the cost of an RPCA solution.

    Arguments:
        Y       -- (ndarray)    the low-rank component
        Z       -- (ndarray)    the sparse component
        alpha   -- (float>0)    the balancing factor

    Returns:
        total, nuclear_norm, l1_norm -- (list of floats)
    '''
    nuclear_norm = scipy.linalg.svd(Y, full_matrices=False, compute_uv=False).sum()
    
    l1_norm = np.abs(Z).sum()
    
    return nuclear_norm + alpha * l1_norm, nuclear_norm, l1_norm


def robust_pca(X, alpha=None, max_iter=100, verbose=False):
    '''ADMM solver for robust PCA.

    min_Y  ||Y||_* + alpha * ||X-Y||_1

    Arguments:

        X        -- (ndarray) input data (d-by-n)
        alpha    -- (float>0) weight of the l1 penalty
                    if unspecified, defaults to 1.0 / sqrt(max(d, n))

    Returns:
        Y        -- (ndarray) low-rank component of X
        Z        -- (ndarray) sparse component of X
        diags    -- (dict)    diagnostic output
    '''
    
    RHO_MIN      = 1e-2
    RHO_MAX      = 1e6
    MAX_RATIO    = 1e1
    SCALE_FACTOR = 2.0e0
    ABS_TOL      = 1e-4
    REL_TOL      = 1e-3
    
    # update rules:
    #  Y+ <- nuclear_prox(X - Z - W, 1/rho)
    #  Z+ <- l1_prox(X - Y - W, alpha/rho)
    #  W+ <- W + Y + Z - X
    
    # Initialize
    rho = RHO_MIN
    
    Y   = X.copy()
    Z   = np.zeros_like(X)
    W   = np.zeros_like(X)
    
    norm_X = scipy.linalg.norm(X)
    
    if alpha is None:
        alpha = max(X.shape)**(-0.5)

    m   = X.size

    _DIAG = {
         'err_primal': [],
         'err_dual':   [],
         'eps_primal': [],
         'eps_dual':   [],
         'rho':        []
    }
    
    for t in range(max_iter):
        Y = nuclear_prox(X - Z - W, 1.0/  rho)
        Z_old = Z.copy()
        Z = l1_prox(X - Y - W, alpha /  rho)
        
        residual_pri  = Y + Z - X
        residual_dual = Z - Z_old
        
        res_norm_pri  = scipy.linalg.norm(residual_pri)
        res_norm_dual = rho * scipy.linalg.norm(residual_dual)
        
        W = W + residual_pri
        
        eps_pri  = np.sqrt(m) * ABS_TOL + REL_TOL * max(scipy.linalg.norm(Y), scipy.linalg.norm(Z), norm_X)
        eps_dual = np.sqrt(m) * ABS_TOL + REL_TOL * scipy.linalg.norm(W)
        
        _DIAG['eps_primal'].append(eps_pri)
        _DIAG['eps_dual'  ].append(eps_dual)
        _DIAG['err_primal'].append(res_norm_pri)
        _DIAG['err_dual'  ].append(res_norm_dual)
        _DIAG['rho'       ].append(rho)
        
        if res_norm_pri <= eps_pri and res_norm_dual <= eps_dual:
            break
            
        if res_norm_pri > MAX_RATIO * res_norm_dual and rho * SCALE_FACTOR <= RHO_MAX:
            rho = rho * SCALE_FACTOR
            W   = W / SCALE_FACTOR
            
            
        elif res_norm_dual > MAX_RATIO * res_norm_pri and rho / SCALE_FACTOR >= RHO_MIN:
            rho = rho / SCALE_FACTOR
            W   = W * SCALE_FACTOR
       
    if verbose:
        if t < max_iter - 1:
            print 'Converged in %d steps' % t
        else:
            print 'Reached maximum iterations'
    
    Y = X - Z
    _DIAG['cost'] = robust_pca_cost(Y, Z, alpha)
    
    return (Y, Z, _DIAG)
