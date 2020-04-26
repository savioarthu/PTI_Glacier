import sys
import numpy as np
from operator import mul, sub
# from skimage.util.shape import *
# from skimage.util import pad
from functools import reduce
from math import floor, sqrt, log10
from scipy.sparse.linalg import svds
from skimage.util.shape import *
from operator import mul, sub
from math import floor, sqrt, log10
import sys
from scipy.stats import chi2
from skimage.util import pad
import timeit
from functools import reduce


sigma = 20                 # Noise standard dev.
ksvd_iter = 10

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ APPROXIMATION PURSUIT METHOD : -----------------------------------------#
#------------------------------------------- ORTHOGONAL MATCHING PURSUIT -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

# data = X
# D = dico


def omp(D, data, sparsity=1):
    #max_error = sqrt(((sigma**1.15)**2)*data.shape[0])
    max_coeff = sparsity

    sparse_coeff = np.zeros((D.shape[1], data.shape[1]))
    tot_res = 0
    for i in range(data.shape[1]):
        count = floor((i+1)/float(data.shape[1])*100)
        sys.stdout.write("\r- Sparse coding : Channel : %d%%" % count)
        sys.stdout.flush()

        x = data[:, i]
        res = x
        atoms_list = []
        res_norm = np.linalg.norm(res)
        temp_sparse = np.zeros(D.shape[1])

        while len(atoms_list) < max_coeff:  # and res_norm > max_error:
            proj = D.T.dot(res)
            i_0 = np.argmax(np.abs(proj))
            atoms_list.append(i_0)

            temp_sparse = np.linalg.pinv(D[:, atoms_list]).dot(x)
            res = x - D[:, atoms_list].dot(temp_sparse)
            res_norm = np.linalg.norm(res)

        tot_res += res_norm
        if len(atoms_list) > 0:
            sparse_coeff[atoms_list, i] = temp_sparse
    print('\r- Sparse coding complete.\n')

    return sparse_coeff

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- DICTIONARY METHODS -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


"""def dict_initiate(train_patches, dict_size):
    # dictionary intialization

    # indexes of patches for dictionary elements
    indexes = np.random.random_integers(0, train_patches.size-1, dict_size)
    # each column is a new atom
    dict_init = np.array(train_patches.image[:, indexes])

    # dictionary normalization
    dict_init = dict_init - dict_init.mean()
    temp = np.diag(
        pow(np.sqrt(np.sum(np.multiply(dict_init, dict_init), axis=0)), -1))
    dict_init = dict_init.dot(temp)
    basis_sign = np.sign(dict_init[0, :])
    dict_init = np.multiply(dict_init, basis_sign)

    print('Shape of dictionary : ', str(dict_init.shape) + '\n')
    # cv2.namedWindow('dict', cv2.WINDOW_NORMAL)
    # cv2.imshow('dict',dict_init.astype('double'))

    return dict_init


def dict_update(D, data, matrix_sparse, atom_id):
    indices = np.where(matrix_sparse[atom_id, :] != 0)[0]
    D_temp = D
    sparse_temp = matrix_sparse[:, indices]

    if len(indices) > 1:
        sparse_temp[atom_id, :] = 0

        matrix_e_k = data[:, indices] - D_temp.dot(sparse_temp)
        u, s, v = svds(np.atleast_2d(matrix_e_k), 1)
        D_temp[:, atom_id] = u[:, 0]
        matrix_sparse[atom_id, indices] = s.dot(v)

    return D_temp, matrix_sparse"""

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def k_svd(train_patches, dict_size, sparsity):

    dict_init = dict_initiate(train_patches, dict_size)

    D = dict_init

    # initializing spare matrix
    matrix_sparse = np.zeros((D.T.dot(train_patches.image)).shape)
    num_iter = ksvd_iter
    print('\nK-SVD, with residual criterion.')
    print('-------------------------------')

    for k in range(num_iter):
        print("Stage ", str(k+1), "/", str(num_iter), "...")

        matrix_sparse = omp(D, train_patches.image, sparsity)

        count = 1

        dict_elem_order = np.random.permutation(D.shape[1])

        for j in dict_elem_order:
            r = floor(count/float(D.shape[1])*100)
            sys.stdout.write("\r- Dictionary updating : %d%%" % r)
            sys.stdout.flush()

            D, matrix_sparse = dict_update(D, train_patches.image, matrix_sparse, j)
            count += 1
        print('\r- Dictionary updating complete.\n')

    return D



#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def new_k_svd(matrix_y, algorithm, sigma=10, n_iter=5, learning_ratio=1, approx='yes'):
    ## matrix_y = patches doit etre numpy.ndarray
    print(matrix_y)
    matrix_y = matrix_y.reshape((matrix_y.shape[0],matrix_y.shape[2])) # reshape, on doit avoir un tab 2D et non 3D

    k = int(learning_ratio*matrix_y.shape[1])
    indexes = np.random.random_integers(0, matrix_y.shape[1]-1, k)
    basis = matrix_y[:, indexes]
    basis /= np.sum(basis.T.dot(basis), axis=-1)

    print('Shape of dictionary : ' + str(basis.shape) + '\n')

    phi = basis
    #print(type(phi)) # doit etre numpy.ndarray

    phi_temp = phi
    matrix_sparse = np.zeros((phi.T.dot(matrix_y)).shape)

    print('\nK-SVD, with residual criterion.')
    print('-------------------------------')

    for k in range(n_iter):
        print("Stage " + str(k+1) + "/" + str(n_iter) + "...")

        def sparse_coding(f):
            t = f+1
            sys.stdout.write("\r- Sparse coding : Channel %d" % t)
            sys.stdout.flush()
            return algorithm(phi_temp, matrix_y[:, f], sigma)[0]

        sparse_rep = list(map(sparse_coding, range(matrix_y.shape[1])))
        matrix_sparse = np.array(sparse_rep).T
        count = 1

        updating_range = phi.shape[1]

        for j in range(updating_range):
            r = floor(count/float(updating_range)*100)
            sys.stdout.write("\r- Dictionary updating : %d%%" % r)
            sys.stdout.flush()
            if approx == 'yes':
                phi_temp, matrix_sparse = approx_update(phi_temp, matrix_y, matrix_sparse, j)
            else:
                phi_temp, matrix_sparse = dict_update(phi_temp, matrix_y, matrix_sparse, j)
            count += 1
        print('\r- Dictionary updating complete.\n')

    return phi_temp, matrix_sparse



#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ APPROXIMATION PURSUIT METHOD : -----------------------------------------#
#------------------------------------- MULTI-CHANNEL ORTHOGONAL MATCHING PURSUIT -----------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def single_channel_omp(phi, vect_y, sigma):

    vect_sparse = np.zeros(phi.shape[1])
    res = np.linalg.norm(vect_y)
    atoms_list = []

    while res/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) \
            and len(atoms_list) < phi.shape[1]:
        vect_c = phi.T.dot(vect_y - phi.dot(vect_sparse))
        i_0 = np.argmax(np.abs(vect_c))
        atoms_list.append(i_0)
        vect_sparse[i_0] += vect_c[i_0]

        # Orthogonal projection.
        index = np.where(vect_sparse)[0]
        vect_sparse[index] = np.linalg.pinv(phi[:, index]).dot(vect_y)
        res = np.linalg.norm(vect_y - phi.dot(vect_sparse))

    return vect_sparse, atoms_list


def cholesky_omp(phi, vect_y, sigma):

    vect_sparse = np.zeros(phi.shape[1])
    atoms_list = []
    vect_alpha = phi.T.dot(vect_y)
    res = vect_y
    l = np.ones((1, 1))
    count = 1

    while np.linalg.norm(res)/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) \
            and len(atoms_list) < vect_sparse.shape[0]:

        c = phi.T.dot(res)
        i_0 = np.argmax(np.abs(c))

        if count > 1:
            w = np.linalg.solve(l, phi[:, atoms_list].T.dot(phi[:, i_0]))
            l = np.insert(l, l.shape[1], 0, axis=1)
            l = np.insert(l, l.shape[0], np.append(w.T, sqrt(1 - np.linalg.norm(w))), axis=0)

        atoms_list.append(i_0)
        vect_sparse[atoms_list] = np.linalg.solve(l.dot(l.T), vect_alpha[atoms_list])
        res = vect_y - phi[:, atoms_list].dot(vect_sparse[atoms_list])
        count += 1

    return vect_sparse, atoms_list


def batch_omp(phi, vect_y, sigma):

    #Initial values
    vect_alpha = phi.T.dot(vect_y)
    epsilon = pow(np.linalg.norm(vect_y), 2)
    matrix_g = phi.T.dot(phi)

    atoms_list = []
    l = np.ones((1, 1))
    vect_sparse = np.zeros(phi.shape[1])
    delta = 0
    count = 1

    while np.linalg.norm(epsilon)/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) \
            and len(atoms_list) < phi.shape[1]:
        i_0 = np.argmax(np.abs(vect_alpha))

        if count > 1:
            w = np.linalg.solve(l, phi.T[atoms_list].dot(phi[:, i_0]))
            l = np.insert(l, l.shape[0], 0, axis=0)
            l = np.insert(l, l.shape[1], 0, axis=1)
            l[-1, :] = np.append(w.T, sqrt(1 - np.linalg.norm(w)))

        atoms_list.append(i_0)
        vect_sparse[atoms_list] = np.linalg.solve(l.dot(l.T), vect_alpha[atoms_list])
        vect_beta = matrix_g[:, atoms_list].dot(vect_sparse[atoms_list])
        vect_alpha = phi.T.dot(vect_y) - vect_beta
        epsilon += - vect_sparse[atoms_list].T.dot(vect_beta[atoms_list]) + delta
        delta = vect_sparse[atoms_list].T.dot(vect_beta[atoms_list])
        count += 1

    return vect_sparse, atoms_list


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- DICTIONARY UPDATING METHODS -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def dict_update(phi, matrix_y, matrix_sparse, k):
    indexes = np.where(matrix_sparse[k, :] != 0)[0]
    phi_temp = phi
    sparse_temp = matrix_sparse

    if len(indexes) > 0:
        phi_temp[:, k][:] = 0

        matrix_e_k = matrix_y[:, indexes] - phi_temp.dot(sparse_temp[:, indexes])
        u, s, v = svds(np.atleast_2d(matrix_e_k), 1)

        phi_temp[:, k] = u[:, 0]
        sparse_temp[k, indexes] = np.asarray(v)[0] * s[0]
    return phi_temp, sparse_temp


def approx_update(phi, matrix_y, matrix_sparse, k):
    indexes = np.where(matrix_sparse[k, :] != 0)[0]
    phi_temp = phi

    if len(indexes) > 0:
        phi_temp[:, k] = 0
        vect_g = matrix_sparse[k, indexes].T
        vect_d = (matrix_y - phi_temp.dot(matrix_sparse))[:, indexes].dot(vect_g)
        vect_d /= np.linalg.norm(vect_d)
        vect_g = (matrix_y - phi_temp.dot(matrix_sparse))[:, indexes].T.dot(vect_d)
        phi_temp[:, k] = vect_d
        matrix_sparse[k, indexes] = vect_g.T
    return phi_temp, matrix_sparse