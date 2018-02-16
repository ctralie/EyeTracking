import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
import matplotlib.pyplot as plt

def getSSM(X):
    """
    Compute a Euclidean self-similarity image between a set of points
    :param X: An Nxd matrix holding the d coordinates of N points
    :return: An NxN self-similarity matrix
    """
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    return D

def getLaplacianEigs(A, NEigs):
    DEG = sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    w, v = slinalg.eigsh(L, k=NEigs, sigma = 0, which = 'LM')
    return (w, v, L)

def getLaplacianEigsDense(A, NEigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG.toarray() - A
    w, v = linalg.eigh(L)
    return (w[0:NEigs], v[:, 0:NEigs], L)
