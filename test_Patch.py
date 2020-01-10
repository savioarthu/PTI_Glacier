from Patch_collection import Patch_collection
import cv2
import numpy as np

paths = ['./Photos/2007060208_cam01.jpg', './Photos/2007060608_cam01.jpg',
		 './Photos/2007060908_cam01.jpg', './Photos/2007061708_cam01.jpg']
paths_chat = ['./Photos_test/chat1.jpg', './Photos_test/chat2.jpg']


images = [cv2.imread(path) for path in paths]


coll = Patch_collection(images, nb=10, size_patch=600)

# coll.select_patches(nb_iter=3, threshold=0.01, fill=False)
print(0)
coll.grid_patches()
print(1)
image = coll.reconstruct_image()
print(2)
cv2.imshow('image reconstructed', image)

# for patch in coll.patches:
# 	patch.show(corrected=True)

while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
