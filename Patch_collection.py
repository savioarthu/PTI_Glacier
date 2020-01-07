from Patch import Patch
from math import sqrt


class Patch_collection():
	def __init__(self, im, nb=10, size_patch=20):
		self.size = size_patch
		self.nb = nb
		self.im = im
		self.patches = []

	def _diff_hists(self, patch1, patch2):
		hists2 = patch2.hists
		hists1 = patch1.hists
		diff = 0
		for hist1, hist2 in zip(hists1, hists2):
			for val1, val2 in zip(hist1, hist2):
				diff += (val1 - val2)** 2
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
				print(self._diff_hists(new_patch, patch))
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
		print(nb_iter)
