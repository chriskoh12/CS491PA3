import pca
import os
import numpy as np
import matplotlib.pyplot

#def compress_images(DATA,k):

def load_data(input_dir):
    #print(input_dir + os.listdir(input_dir)[0])
    A = np.asarray(matplotlib.pyplot.imread(input_dir + os.listdir(input_dir)[0]))
    for i in range(1, len(os.listdir(input_dir))):
        A = np.asarray(matplotlib.pyplot.imread(input_dir + os.listdir(input_dir)[i]))
    print(A.flatten())


#print(matplotlib.pyplot.imread('Data/Train/00001_930831_fa_a.pgm'))

load_data('Data/Train/')