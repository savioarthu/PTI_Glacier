from random import randint
import numpy as np
import cv2

colors = [0, 1, 2]


class Patch():
	def __init__(self, im, x=None, y=None, size_patch=20):
		if type(im) is list:
			self.nb_images = len(im)
			self.image_index = randint(0, self.nb_images - 1)
			self.im = im[self.image_index]
		else:
			self.nb_images, self.image_index = 0, 0
			self.im = im

		self.nb_images = len(im) if type(im) is list else 0

		self.size = size_patch
		im_size = self.im.shape

		self.x = x if ((x is not None) and (x >= 0) and (
			x <= im_size[0] - self.size)) else randint(0, im_size[0] - self.size)
		self.y = y if ((y is not None) and (y >= 0) and (
			y <= im_size[1] - self.size)) else randint(0, im_size[1] - self.size)

		self.im = self.im[self.x:self.x +
						  self.size, self.y:self.y + self.size, :]
		self.image_to_patch()

		self.hists = [np.histogram(self.im[color], bins=np.arange(-2, 2, 0.1), density=True)[
			0] for color in colors]

	def image_to_patch(self):
		self.mean = [self.im[:, :, color].mean() for color in colors]
		self.std = [self.im[:, :, color].std() for color in colors]
		self.im = [np.reshape((self.im[:, :, color] - self.mean[color]) /
							  self.std[color], (1, self.size ** 2)) for color in colors]

	def patch_to_image(self, uncentered=True):
		if uncentered:
			image = [[pixR * self.std[0] + self.mean[0], pixG * self.std[1] + self.mean[1], pixB *
					 self.std[2] + self.mean[2]] for pixR, pixG, pixB in zip(self.im[0], self.im[1], self.im[2])]
		else:
			print(np.shape(self.im))
			image = [[pixR, pixG, pixB] for pixR, pixG, pixB in zip(self.im[0], self.im[1], self.im[2])]
			print(image)
		return np.reshape(image, (self.size, self.size, 3))

	def show(self, im=None):
		#print(f"x: {self.x}\ty: {self.y}\timage: {self.image_index}")
		reshaped_im = self.patch_to_image(False)
		if im is not None:
			base_im = im[self.image_index][self.x:self.x +
										   self.size, self.y:self.y + self.size, :] / 256
		else:
			base_im = self.patch_to_image()
		# image = np.concatenate(
		# 	(reshaped_im, base_im), axis=1)
		image = base_im
		print(np.shape(self.im))
		print(self.im)
		print(np.shape(base_im))
		print(base_im)
		#cv2.imshow(f'patch{self.x}_{self.y}_image{self.image_index}', image)
