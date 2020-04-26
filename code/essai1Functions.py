#!/usr/bin/python3
# 2020 - IMSI - Projet Traitement des images
# Arthur Savio - Félix Cloup - Yoann Ruiz - Maxime Lesaffre - Alexis Teste
# Fonctions 



# Import des librairies nécessaires
import cv2
from time import time
import numpy as np

from skimage.util.shape import *
from operator import mul, sub
from math import floor, sqrt, log10
import sys
from scipy.sparse.linalg import svds
from scipy.stats import chi2
from skimage.util import pad
import timeit
from functools import reduce
import matplotlib.pyplot as plt
from random import randint
import scipy.misc


colors = [0, 1, 2]


class Patch():
	def __init__(self, image, x=None, y=None, size_patch=20):
		if type(image) is list:
			nb_images = len(image)
			self.image_index = randint(0, nb_images - 1)
			self.image = image[self.image_index]
		else:
			self.image_index = 0
			self.image = image

		self.image_shape = np.shape(self.image)

		self.size = size_patch
		self.x = x if ((x is not None) and (x >= 0) and (
			x <= self.image_shape[0] - self.size)) else randint(0, self.image_shape[0] - self.size)
		self.y = y if ((y is not None) and (y >= 0) and (
			y <= self.image_shape[1] - self.size)) else randint(0, self.image_shape[1] - self.size)

		self.image = self.image[self.x:self.x +
								self.size, self.y: self.y + self.size, :]
		self.image_to_vector()

		self.hists = [np.histogram(self.vector[color], bins=np.arange(-2, 2, 0.1), density=True)[
			0] for color in colors]

	def image_to_vector(self):
		vector = self.image
		self.mean = [vector[:, :, color].mean() for color in colors]
		self.std = [vector[:, :, color].std() for color in colors]
		self.vector = [np.reshape((vector[:, :, color] - self.mean[color]) /
								  self.std[color], (1, self.size ** 2)) for color in colors]

	def vector_to_image(self):
		image = [[(pixR * self.std[0] + self.mean[0]) / 256,
				  (pixG*self.std[1] + self.mean[1]) / 256,
				  (pixB*self.std[2] + self.mean[2]) / 256]
				 for pixR, pixG, pixB in zip(self.vector[0][0], self.vector[1][0], self.vector[2][0])]
		self.image = np.reshape(image, (self.size, self.size, 3))

	def show(self, method='plt'):
		print(f"x: {self.x}\ty: {self.y}\timage: {self.image_index}")
		self.vector_to_image()
		if method == 'cv2':
			cv2.imshow(
				f'patch{self.x}_{self.y}_image{self.image_index}', self.image)
		else:
			plt.title(f'patch{self.x}_{self.y}_image{self.image_index}')
			plt.imshow(self.image)
			plt.show()

	def show_hists(self):
		plt.figure(figsize=(20, 5))

		plt.subplot(1, 3, 1)
		plt.bar(np.arange(39), self.hists[0], color='red')
		plt.subplot(1, 3, 2)
		plt.bar(np.arange(39), self.hists[1], color='blue')
		plt.subplot(1, 3, 3)
		plt.bar(np.arange(39), self.hists[2], color='green')

		plt.show()

	def encode(self, dicts, sparsity=1):
		self.vector = [omp(dicts[color], self.vector[color], sparsity)
					   for color in colors]

	def decode(self, dicts):
		self.vector = [np.matmul(dicts[color], self.vector[color])
					   for color in colors]


class Patch_collection():
	def __init__(self, im, nb=1000, size_patch=32):
		self.size = size_patch
		self.nb = nb
		self.im = im
		self.size_image = np.shape(self.im)
		self.patches = []

	def _diff_hists(self, patch1, patch2):
		hists2 = patch2.hists
		hists1 = patch1.hists
		diff = 0
		for hist1, hist2 in zip(hists1, hists2):
			for val1, val2 in zip(hist1, hist2):
				diff += (val1 - val2) ** 2
		return (sqrt(diff) / self.size)

	def _pick_patches(self):
		for i in range(self.nb - len(self.patches)):
			self.patches.append(Patch(self.im, size_patch=self.size))

	def _clear_patches(self, threshold=0.1):
		patches = self.patches
		self.patches = [patches[0]]
		for new_patch in patches:
			is_different = True
			for patch in self.patches:
				if self._diff_hists(new_patch, patch) < threshold:
					is_different = False
					break
			if is_different:
				self.patches.append(new_patch)

	def select_patches(self, nb_iter=10, fill=False, threshold=0.1):
		while nb_iter > 0 and len(self.patches) < self.nb:
			self._pick_patches()
			self._clear_patches(threshold)
			nb_iter -= 1
		if fill:
			self._pick_patches()
		self.sort_patches()
		"""for patch in self.patches:
			patch.show()
			patch.show_hists()"""

	def sort_patches(self):
		self.patches.sort(key=lambda patch: sum(patch.std), reverse=True)

	def grid_patches(self, im=None):
		if im is None:
			im = self.im
		if type(im) is list:
			im = im[0]

		self.height, self.length, _ = im.shape
		offset_patches = int(self.size * 0.75)
		self.patches = []
		for x in range(0, self.height - self.size, offset_patches):
			for y in range(0, self.length - self.size, offset_patches):
				self.patches.append(Patch(im, x, y, self.size))
			self.patches.append(
				Patch(im, x, self.length - self.size, self.size))

		for y in range(0, self.length - self.size, offset_patches):
			self.patches.append(
				Patch(im, self.height - self.size, y, self.size))
		self.patches.append(Patch(im, self.height - self.size,
								  self.length - self.size, self.size))

	def reconstruct_image(self):
			# [r, g, b, counter]
			image = np.full(
				(self.size_image[1], self.size_image[2], 4), 0, dtype=np.float64)
			for patch in self.patches:
				patch.vector_to_image()
				for x in range(self.size):
					for y in range(self.size):
						image[patch.x + x, patch.y + y][:3] += patch.image[x, y]
						image[patch.x + x, patch.y + y][3] += 1
			image_return = np.zeros(
				(self.size_image[1], self.size_image[2], 3), dtype=np.float64)
			for x in range(self.size_image[1]):
				for y in range(self.size_image[2]):
					#image_return[x, y] = image[x, y][:3] / image[x, y][3]
					image_return[x, y] = np.divide(image[x, y][:3], image[x, y][3], where=image[x, y][3]!=0)

			return image_return

	def encode(self, dicts, sparsity=1):
		for patch in self.patches:
			patch.encode(dicts, sparsity)

	def decode(self, dicts):
		for patch in self.patches:
			patch.decode(dicts)

	def split_channels(self):
		R = []
		G = []
		B = []
		for patch in self.patches:
			R.append(patch.vector[0])
			G.append(patch.vector[1])
			B.append(patch.vector[2])
		return R, G, B




def patch_matrix_windows(img, stride):
	# we return an array of patches(patch_size X num_patches)
	patches = view_as_windows(img, window_shape, step=stride)  # shape = [patches in image row,patches in image col,rows in patch,cols in patch]
	# size of cond_patches = patch size X number of patches
	cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
	for i in range(patches.shape[0]):
		for j in range(patches.shape[1]):
			cond_patches[:, j+patches.shape[1]*i] = np.concatenate(patches[i, j], axis=0)
	return cond_patches, patches.shape



#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def new_k_svd(matrix_y, algorithm, sigma=10, n_iter=5, learning_ratio=1, approx='yes'):
	## matrix_y = patches doit etre numpy.ndarray
	new_patches = np.array(matrix_y.vector)
	print(new_patches.shape)
	new_patches = new_patches.reshape((new_patches.shape[0],new_patches.shape[2])) 
	matrix_y = new_patches#matrix_y.reshape((matrix_y.shape[0],matrix_y.shape[2])) # reshape, on doit avoir un tab 2D et non 3D

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


class KSVD():
	def __init__(self,
				 collection_aprentissage=None, collection_aprentissage_im=None, collection_aprentissage_nb=1000, collection_size=32,
				 aprentissage_nb_iter=10, aprentissage_threshold=0.1, aprentissage_fill=False,
				 collection_debruitage=None, collection_debruitage_im=None, collection_debruitage_nb=1000):

		if collection_aprentissage is not None:
			self.collection_aprentissage = collection_aprentissage
		else:
			assert collection_aprentissage_im is not None
			self.collection_aprentissage = Patch_collection(
				collection_aprentissage_im, collection_aprentissage_nb, collection_size
			)
			self.collection_aprentissage.select_patches()

		self.aprentissage_nb_iter = aprentissage_nb_iter
		self.aprentissage_threshold = aprentissage_threshold
		self.aprentissage_fill = aprentissage_fill

		if collection_debruitage is not None:
			self.collection_debruitage = collection_debruitage
		else:
			assert collection_debruitage_im is not None
			self.collection_debruitage = Patch_collection(
				collection_debruitage_im, collection_debruitage_nb, collection_size
			)
			self.collection_debruitage.grid_patches()

		self.create_dict()

	def create_dict(self):
		self.collection_aprentissage.select_patches(
			self.aprentissage_nb_iter, fill=self.aprentissage_fill, threshold=self.aprentissage_threshold)
		patches = self.collection_aprentissage.patches

		self.dicts = [new_k_svd(patches[color], single_channel_omp) for color in colors]

		#self.dicts.sort(key=lambda patch: sum(np.std(patch, axis=1)))

	def encode(self):
		self.collection_debruitage.encode(self.dicts)

	def decode(self):
		self.collection_debruitage.decode(self.dicts)

	def denoise(self, show_output=False, save_output=None):
		self.encode()
		self.decode()
		if show_output:
			img = self.collection_debruitage.reconstruct_image()
			print(img)
			cv2.imshow('denoised image', img)
		if save_output:
			img = self.collection_debruitage.reconstruct_image()
			print(img)
			cv2.imwrite("/test.jpg", img)
			scipy.misc.imsave('test.jpg', img)