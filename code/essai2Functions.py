#!/usr/bin/python3
# 2020 - IMSI - Projet Traitement des images
# Arthur Savio - Félix Cloup - Yoann Ruiz - Maxime Lesaffre - Alexis Teste
# Fonctions 



# Import des librairies nécessaires
import cv2
from time import time
import numpy as np

from operator import mul, sub
from skimage.util.shape import *
from skimage.util import pad
from functools import reduce
from math import floor, sqrt, log10
from scipy.sparse.linalg import svds
import timeit
import sys
from scipy.stats import chi2
from matplotlib import pyplot as plt


## Parameters
sigma = 10
patch_size = 8
window_shape = (patch_size, patch_size)    # Patches' shape
window_stride = 8                  # Patches' step
dict_ratio = 0.1            # Ratio for the dictionary (training set).
num_dict = 300
ksvd_iter = 10
max_sparsity = 1
max_resize_dim = 512
dict_train_blocks = 65000


def patch_matrix_windows(img, stride):
	# we return an array of patches(patch_size X num_patches)
	patches = view_as_windows(img, window_shape, step=stride)  # shape = [patches in image row,patches in image col,rows in patch,cols in patch]
	# size of cond_patches = patch size X number of patches
	cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
	for i in range(patches.shape[0]):
		for j in range(patches.shape[1]):
			cond_patches[:, j+patches.shape[1]*i] = np.concatenate(patches[i, j], axis=0)
	return cond_patches, patches.shape


def reconstruct_image(patch_final, noisy_image):
	img_out = np.zeros(noisy_image.shape)
	weight = np.zeros(noisy_image.shape)
	num_blocks = noisy_image.shape[0] - patch_size + 1
	for l in range(patch_final.shape[1]):
		i, j = divmod(l, num_blocks)
		temp_patch = patch_final[:, l].reshape(window_shape)
		# img_out[i, j] = temp_patch[1, 1]
		img_out[i:(i+patch_size), j:(j+patch_size)] = img_out[i:(i+patch_size), j:(j+patch_size)] + temp_patch
		weight[i:(i+patch_size), j:(j+patch_size)] = weight[i:(i+patch_size), j:(j+patch_size)] + np.ones(window_shape)

	# img_out = img_out/weight
	img_out = (noisy_image+0.034*sigma*img_out)/(1+0.034*sigma*weight)

	return img_out

def image_reconstruction_windows(mat_shape, patch_mat, patch_sizes, step):
    img_out = np.zeros(mat_shape)
    for l in range(patch_mat.shape[1]):
        i, j = divmod(l, patch_sizes[1])
        temp_patch = patch_mat[:, l].reshape((patch_sizes[2], patch_sizes[3]))
        img_out[i*step:(i+1)*step, j*step:(j+1)*step] = temp_patch[:step, :step].astype(int)
    return img_out


def omp(D, data, sparsity):
    max_error = sqrt(((sigma**1.15)**2)*data.shape[0])
    max_coeff = sparsity

    sparse_coeff = np.zeros((D.shape[1],data.shape[1]))
    tot_res = 0
    for i in range(data.shape[1]):
        count = floor((i+1)/float(data.shape[1])*100)
        sys.stdout.write("\r- Code parcimonieux : Channel : %d%%" % count)
        sys.stdout.flush()
        
        x = data[:,i]
        res = x
        atoms_list = []
        res_norm = np.linalg.norm(res)
        temp_sparse = np.zeros(D.shape[1])

        while len(atoms_list) < max_coeff:
            proj = D.T.dot(res)
            i_0 = np.argmax(np.abs(proj))
            atoms_list.append(i_0)

            temp_sparse = np.linalg.pinv(D[:,atoms_list]).dot(x)
            res = x - D[:,atoms_list].dot(temp_sparse)
            res_norm = np.linalg.norm(res)

        tot_res += res_norm
        if len(atoms_list) > 0:
            sparse_coeff[atoms_list, i] = temp_sparse
    print ('\r- Codage parcimonieux complet.\n')

    return sparse_coeff

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

		# Projection orthogonale
		index = np.where(vect_sparse)[0]
		vect_sparse[index] = np.linalg.pinv(phi[:, index]).dot(vect_y)
		res = np.linalg.norm(vect_y - phi.dot(vect_sparse))

	return vect_sparse, atoms_list


def dict_initiate(train_patches, dict_size):
	# Init dictionnaire
	
	indexes = np.random.random_integers(0, train_patches.shape[1]-1, dict_size)
	dict_init = np.array(train_patches[:, indexes])

	# Normalise dictionnaire
	dict_init = dict_init - dict_init.mean()
	temp = np.diag(pow(np.sqrt(np.sum(np.multiply(dict_init,dict_init),axis=0)), -1))
	dict_init = dict_init.dot(temp)
	basis_sign = np.sign(dict_init[0,:])
	dict_init = np.multiply(dict_init, basis_sign)

	print( 'Taille dictionnaire : ' , str(dict_init.shape) + '\n')

	return dict_init

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


def k_svd(matrix_y, sigma, n_iter, approx='yes'):
	phi = dict_initiate(matrix_y, num_dict)
	phi_temp = phi
	matrix_sparse = np.zeros((phi.T.dot(matrix_y)).shape)

	print('\nK-SVD')
	print('-------------------------------')

	for k in range(n_iter):
		print("Stage " + str(k+1) + "/" + str(n_iter) + "...")

		def sparse_coding(f):
			t = f+1
			sys.stdout.write("\r- Sparse coding : Channel %d" % t)
			sys.stdout.flush()
			return single_channel_omp(phi_temp, matrix_y[:, f], sigma)[0]

		sparse_rep = list(map(sparse_coding, range(matrix_y.shape[1])))
		matrix_sparse = np.array(sparse_rep).T
		count = 1

		updating_range = phi.shape[1]

		for j in range(updating_range):
			r = floor(count/float(updating_range)*100)
			sys.stdout.write("\r- Mise à jour du dictionnaire : %d%%" % r)
			sys.stdout.flush()
			if approx == 'yes':
				phi_temp, matrix_sparse = approx_update(phi_temp, matrix_y, matrix_sparse, j)
			else:
				phi_temp, matrix_sparse = dict_update(phi_temp, matrix_y, matrix_sparse, j)
			count += 1
		print('\r- Mise à jour complète.\n')

	return phi_temp, matrix_sparse


def psnr(original_image, approximation_image):
	return 20*log10(np.amax(original_image)) - 10*log10(pow(np.linalg.norm(original_image - approximation_image), 2)
														/ approximation_image.size)


def plot_patches(patches, w=20, h=20):
	fig=plt.figure(figsize=(15, 15))
	for i in range(patches.shape[0]):
		plt.subplot(10,10, i+1)
		plt.imshow(patches[i].reshape(patches[i].shape[0],1))
	plt.xticks(())
	plt.yticks(())
	plt.show()


def denoising(noisy_image, learning_image, dict_size, sparsity):

	# Padding images
	padded_image = pad(learning_image, pad_width=window_shape, mode='symmetric')
	padded_noisy_image = pad(noisy_image, pad_width=window_shape, mode='symmetric')
	poss_patches = (learning_image.shape[0]-patch_size + 1) * (learning_image.shape[1]-patch_size +1)
	stride = floor(poss_patches/dict_train_blocks)
	if stride<1:
		stride = 1
	stride = 2

	# Entraînement dictionnaire
	train_patches, train_patches_shape = patch_matrix_windows(padded_image, stride)
	#plot_patches(train_patches)
	train_data_mean = train_patches.mean()
	train_patches = train_patches - train_data_mean
	dict_final, sparse_init = k_svd(train_patches, sigma, ksvd_iter)
	#cv2.namedWindow('dict', cv2.WINDOW_NORMAL)
	#cv2.imshow('dict',dict_final.astype('double'))

	## Reconstruction image
	noisy_patches, noisy_patches_shape = patch_matrix_windows(padded_noisy_image, stride=1)
	data_mean = noisy_patches.mean()
	noisy_patches = noisy_patches - data_mean
	sparse_final = omp(dict_final, noisy_patches, max_sparsity)
	patches_approx = dict_final.dot(sparse_final) + data_mean
	padded_denoised_image = reconstruct_image(patches_approx, padded_noisy_image)
	shrunk_0, shrunk_1 = tuple(map(sub, padded_denoised_image.shape, window_shape))
	denoised_image = np.abs(padded_denoised_image)[window_shape[0]:shrunk_0, window_shape[1]:shrunk_1]

	return denoised_image