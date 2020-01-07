from Patch_collection import Patch_collection
import cv2

paths = ['./Photos/2007060208_cam01.jpg', './Photos/2007060608_cam01.jpg', './Photos/2007060908_cam01.jpg', './Photos/2007061708_cam01.jpg']
paths_chat = ['./Photos_test/chat1.jpg', './Photos_test/chat2.jpg']


images = [cv2.imread(path) for path in paths]


coll = Patch_collection(images, nb=10, size_patch=400)
coll.select_patches(nb_iter=10, threshold=0.015, fill=False)

for patch in coll.patches[:10]:
	patch.show(im=images)

while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
