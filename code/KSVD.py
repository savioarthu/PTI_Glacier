import numpy as np
import cv2

from apprentissage import k_svd, new_k_svd, single_channel_omp
from Patch_collection import Patch_collection

colors = [0, 1, 2]


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
            nb_iter = self.aprentissage_nb_iter, fill=self.aprentissage_fill, threshold=self.aprentissage_threshold)
        patches = self.collection_aprentissage.patches

        self.dicts = [new_k_svd(patches[color], single_channel_omp) for color in colors]

        self.dicts.sort(key=lambda patch: sum(np.std(patch, axis=1)))

    def encode(self):
        self.collection_debruitage.encode(self.dicts)

    def decode(self):
        self.collection_debruitage.decode(self.dicts)

    def denoise(self, show_output=False, save_output=None):
        self.encode()
        self.decode()
        if show_output:
            img = self.collection_debruitage.reconstruct_image()
            cv2.imshow('denoised image', img)
        if save_output:
            img = self.collection_debruitage.reconstruct_image()
            cv2.imwrite(save_output, img)
