from KSVD import KSVD
import cv2
from Patch_collection import Patch_collection

from time import time

start = time()
paths = ['./Photos/2007060208_cam01.jpg', './Photos/2007060608_cam01.jpg',
         './Photos/2007060908_cam01.jpg', './Photos/2007061708_cam01.jpg']
paths_chat = ['./Photos_test/chat1.jpg', './Photos_test/chat2.jpg']


images = [cv2.imread(path) for path in paths_chat]


coll_learn = Patch_collection(images, nb=100, size_patch=32)
coll_denoise = Patch_collection(images[0], nb=100, size_patch=32)


coll_denoise.grid_patches()
coll_learn.select_patches()

ksvd = KSVD(100, 1, collection_aprentissage=coll_learn,
            collection_debruitage=coll_denoise)


end = time()
print("Time :", round(end-start, 3), "seconds")
