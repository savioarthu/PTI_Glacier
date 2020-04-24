from KSVD import KSVD
import cv2
from Patch_collection import Patch_collection
from apprentissage import new_k_svd, single_channel_omp
from time import time
import numpy as np


start = time()
paths = ['./Photos/2007060208_cam01.jpg', './Photos/2007060608_cam01.jpg',
         './Photos/2007060908_cam01.jpg', './Photos/2007061708_cam01.jpg']
paths_chat = ['./Photos_test/chat1.jpg', './Photos_test/chat2.jpg']


images = [cv2.imread(path) for path in paths_chat]
coll_learn = Patch_collection(images, nb=100, size_patch=32)


R, G, B = coll_learn.split_channels()

D_R, Gamma_R = new_k_svd(np.array(R), single_channel_omp)
D_G, Gamma_G = new_k_svd(np.array(G), single_channel_omp)
D_B, Gamma_B = new_k_svd(np.array(B), single_channel_omp)

## Print result
print(Gamma_R)
print(D_R)

# Marche sans passer par le collection patch

coll_learn = Patch_collection(images, nb=100, size_patch=32)
coll_denoise = Patch_collection(images[0], nb=100, size_patch=32)


coll_denoise.grid_patches()
coll_learn.select_patches()

## Maintennant on veut que ça passe avec la collection... il faut adapter le new_ksvd pour que ça fonctionne

#ksvd = KSVD(collection_aprentissage=coll_learn, collection_debruitage=coll_denoise)


## Ensuite affichage regarder si fonctionne
# ksvd.denoise(show_output=True)

end = time()
print("Time :", round(end-start, 3), "seconds")