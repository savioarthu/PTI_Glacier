from Patch import Patch
from math import sqrt
import numpy as np


class Patch_collection():
    def __init__(self, im, nb=10, size_patch=20):
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

    def grid_patches(self, im=None):
        if im is None:
            im = self.im
        if type(im) is list:
            im = im[0]

        self.height, self.length, _ = im.shape
        self.offset_patches = int(self.size * 0.75)
        self.patches = []
        for x in range(0, self.height - self.size, self.offset_patches):
            for y in range(0, self.length - self.size, self.offset_patches):
                self.patches.append(Patch(im, x, y, self.size))
            self.patches.append(
                Patch(im, x, self.length - self.size, self.size))

        for y in range(0, self.length - self.size, self.offset_patches):
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
        print("sum done")
        image_return = np.zeros(
            (self.size_image[1], self.size_image[2], 3), dtype=np.float64)
        for x in range(self.size_image[1]):
            for y in range(self.size_image[2]):
                image_return[x, y] = image[x, y][:3] / image[x, y][3]
        print("divide done")
        return image_return
