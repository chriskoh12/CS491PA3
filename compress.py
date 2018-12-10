import pca
import os
import numpy as np
import matplotlib.pyplot

def compress_images(DATA,k):
    Z = pca.compute_Z(DATA)
    #print(Z)
    COV = pca.compute_covariance_matrix(Z)
    #print(COV)
    L, PCS = pca.find_pcs(COV)
    #print(L)
    #print(PCS)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    PCS = PCS[:, :k]
    X_compressed = Z_star @ PCS.transpose()
    X_compressed = X_compressed.transpose()
    #print(X_compressed)
    if not os.path.exists('Output'):
        os.makedirs('Output')
    for i in range(0, 869):
        matplotlib.pyplot.imsave("Output/output" + str(i) + ".png", X_compressed[i].reshape(60,48), cmap='gray')


def load_data(input_dir):
    #print(input_dir + os.listdir(input_dir)[0]) //name of file
    A = np.asarray(matplotlib.pyplot.imread(input_dir + os.listdir(input_dir)[0])).flatten()
    A = np.reshape(A, (-1, 1))
    for i in range(1, len(os.listdir(input_dir))):
        B = np.asarray(matplotlib.pyplot.imread(input_dir + os.listdir(input_dir)[i])).flatten()
        B = np.reshape(B, (-1, 1))
        A = np.append(A, B, axis=1)
    #print(A) print the matrix with all image values
    return A.astype(np.float)

DATA = load_data('Data/Train/')
#print(DATA)

result = compress_images(DATA, 10)
#print(result)