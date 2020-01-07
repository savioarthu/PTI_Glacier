from random import randint
import numpy as np
import cv2


class Patch():
	def __init__(self, im, x=None, y=None, size_patch=20):
		self.nb_images = len(im)
		self.image_index = randint(0, self.nb_images - 1)

		self.size = size_patch
		im_size = im[0].shape

		self.x = x if (x is not None) and (x >= 0) and (
			x <= im_size[0] - self.size) else randint(0, im_size[0] - self.size)
		self.y = y if (y is not None) and (y >= 0) and (
			y <= im_size[1] - self.size) else randint(0, im_size[1] - self.size)

		self.im = im[self.image_index][self.x:self.x + self.size, self.y:self.y + self.size, :]
		self.im = np.reshape(self.im, (self.size ** 2, 3))
		self.im = (self.im - self.im.mean()) / self.im.std()

		self.hists = [np.histogram(self.im[:, i], bins=np.arange(-2, 2, 0.1), density=True)[
			0] for i in range(3)]

	def show(self, im=None):
		print(f"x: {self.x}\ty: {self.y}\timage: {self.image_index}")
		reshaped_im = np.reshape(self.im, (self.size, self.size, 3))
		if im is not None:
			base_im = im[self.image_index][self.x:self.x + self.size, self.y:self.y + self.size,:] / 256
			image = np.concatenate((reshaped_im, base_im), axis=1)
		else:
			image = reshaped_im
		cv2.imshow(f'patch{self.x}_{self.y}_image{self.image_index}', image)