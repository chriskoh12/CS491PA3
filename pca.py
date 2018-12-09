import numpy as np


def compute_Z(X, centering=True, scaling=False):
    X = X.astype(np.float)
    if centering:
        Y = np.mean(X, axis=0)
        for i in range (len(X[0])):
            for j in range (len(X)):
                X[j][i] -= Y[i]
        #print(X)

    if scaling:
        Y = np.std(X, axis=0)
        for i in range (len(X[0])):
            for j in range (len(X)):
                X[j][i] /= Y[i]
    #print("Z matrix is: ")
    #print(X)
    return X


def compute_covariance_matrix(Z):
    #print("Covariance Matrix is: ")
    #print(np.transpose(Z) @ Z)
    return np.transpose(Z) @ Z


def find_pcs(COV):
    L, PCS = np.linalg.eig(COV)
    #print(L) #eigenvalues and vectors before sort
    #print(PCS)
    for i in range(len(L)):
        for j in range(0, (len(L))-i-1):
            if L[j] < L[j+1]:
                L[j], L[j+1] = L[j+1], L[j]
                for k in range(len(PCS)):
                    PCS[k][j], PCS[k][j+1] = PCS[k][j+1], PCS[k][j]
    #print("Eigenvalues and eigenvectors are:")
    #print(L)  #eigenvalues and vectors after sort
    #print(PCS)
    return L, PCS


def project_data(Z, PCS, L, k, var):
    if k == 0:
        eigenTotal = 0
        for i in range(len(L)):
            eigenTotal += L[i]
        #print(eigenTotal)
        temp = 0.0
        while temp < var:
            temp = 0.0
            k += 1
            for i in range(0, k):
                temp += L[i]
            temp /= eigenTotal
        PCS = PCS[:, :k]  # slice the PCS array to keep the first k number of elements
    else:
        PCS = PCS[:, :k] #slice the PCS array to keep the first k number of elements
        #print(PCS) #test if slicing properly
    return Z @ PCS

X = np.array([[1, 1], [2, 7], [3, 3], [4, 4], [5, 5]])
Z = compute_Z(X)
COV = compute_covariance_matrix(Z)
L, PCS = find_pcs(COV)
Z_star = project_data(Z, PCS, L, 2, 0)
#print("Z_star is")
#print(Z_star)
