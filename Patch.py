from random import randint
import numpy as np
import cv2

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

	def show(self, im=None):
		print(f"x: {self.x}\ty: {self.y}\timage: {self.image_index}")
		self.vector_to_image()
		cv2.imshow(
			f'patch{self.x}_{self.y}_image{self.image_index}', self.image)
