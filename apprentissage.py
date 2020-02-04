import sys
import numpy as np
from operator import mul, sub
from skimage.util.shape import *
from skimage.util import pad
from functools import reduce
from math import floor, sqrt, log10
from scipy.sparse.linalg import svds


sigma = 20                 # Noise standard dev.
ksvd_iter = 10

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ APPROXIMATION PURSUIT METHOD : -----------------------------------------#
#------------------------------------------- ORTHOGONAL MATCHING PURSUIT -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

# data = X
# D = dico
def omp(D, data, sparsity=1):
    max_error = sqrt(((sigma**1.15)**2)*data.shape[0])
    max_coeff = sparsity


    sparse_coeff = np.zeros((D.shape[1],data.shape[1]))
    tot_res = 0
    for i in range(data.shape[1]):
        count = floor((i+1)/float(data.shape[1])*100)
        sys.stdout.write("\r- Sparse coding : Channel : %d%%" % count)
        sys.stdout.flush()
        
        x = data[:,i]
        res = x
        atoms_list = []
        res_norm = np.linalg.norm(res)
        temp_sparse = np.zeros(D.shape[1])

        while len(atoms_list) < max_coeff: #and res_norm > max_error:
            proj = D.T.dot(res)
            i_0 = np.argmax(np.abs(proj))
            atoms_list.append(i_0)

            temp_sparse = np.linalg.pinv(D[:,atoms_list]).dot(x)
            res = x - D[:,atoms_list].dot(temp_sparse)
            res_norm = np.linalg.norm(res)

        tot_res += res_norm
        if len(atoms_list) > 0:
            sparse_coeff[atoms_list, i] = temp_sparse
    print('\n',tot_res)
    print ('\r- Sparse coding complete.\n')

    return sparse_coeff

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- DICTIONARY METHODS -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def dict_initiate(train_patches, dict_size):
    # dictionary intialization
    
    indexes = np.random.random_integers(0, train_patches.shape[1]-1, dict_size)   # indexes of patches for dictionary elements
    dict_init = np.array(train_patches[:, indexes])            # each column is a new atom

    # dictionary normalization
    dict_init = dict_init - dict_init.mean()
    temp = np.diag(pow(np.sqrt(np.sum(np.multiply(dict_init,dict_init),axis=0)), -1))
    dict_init = dict_init.dot(temp)
    basis_sign = np.sign(dict_init[0,:])
    dict_init = np.multiply(dict_init, basis_sign)

    print( 'Shape of dictionary : ' , str(dict_init.shape) + '\n')
    # cv2.namedWindow('dict', cv2.WINDOW_NORMAL)
    # cv2.imshow('dict',dict_init.astype('double'))

    return dict_init


def dict_update(D, data, matrix_sparse, atom_id):
    indices = np.where(matrix_sparse[atom_id, :] != 0)[0]
    D_temp = D
    sparse_temp = matrix_sparse[:,indices]

    if len(indices) > 1:
        sparse_temp[atom_id,:] = 0

        matrix_e_k = data[:, indices] - D_temp.dot(sparse_temp)
        u, s, v = svds(np.atleast_2d(matrix_e_k), 1)
        D_temp[:, atom_id] = u[:, 0]
        matrix_sparse[atom_id, indices] = s.dot(v)

    return D_temp, matrix_sparse

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def k_svd(train_patches, dict_size, sparsity):

    dict_init = dict_initiate(train_patches, dict_size)

    D = dict_init

    matrix_sparse = np.zeros((D.T.dot(train_patches)).shape)         # initializing spare matrix
    num_iter = ksvd_iter
    print ('\nK-SVD, with residual criterion.')
    print ('-------------------------------')

    for k in range(num_iter):
        print ("Stage " , str(k+1) , "/" , str(num_iter) , "...")

        matrix_sparse = omp(D, train_patches, sparsity)

        count = 1

        dict_elem_order = np.random.permutation(D.shape[1])

        for j in dict_elem_order:
            r = floor(count/float(D.shape[1])*100)
            sys.stdout.write("\r- Dictionary updating : %d%%" % r)
            sys.stdout.flush()
            
            D, matrix_sparse = dict_update(D, train_patches, matrix_sparse, j)
            count += 1
        print ('\r- Dictionary updating complete.\n')

    return D, matrix_sparse