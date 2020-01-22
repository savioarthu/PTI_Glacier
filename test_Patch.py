from Patch_collection import *
from apprentissage import *
import cv2
import time

start = time.time()
paths = ['./Photos/2007060208_cam01.jpg', './Photos/2007060608_cam01.jpg',
		 './Photos/2007060908_cam01.jpg', './Photos/2007061708_cam01.jpg']
paths_chat = ['./Photos_test/chat1.jpg', './Photos_test/chat2.jpg']


images = [cv2.imread(path) for path in paths]


coll = Patch_collection(images, nb=1000, size_patch=32)

# coll.select_patches(nb_iter=3, threshold=0.01, fill=False)
#print(0)
#coll.grid_patches()
#print(1)
#image = coll.reconstruct_image()
#print(2)
#cv2.imshow('image reconstructed', image)

# for patch in coll.patches:
# 	patch.show(corrected=True)

#while True:
#	if cv2.waitKey(1) & 0xFF == ord('q'):
#		break

#cv2.destroyAllWindows()


# Step 1 - Recover patches
coll.select_patches(nb_iter=3, threshold=0.01, fill=False)
R, G, B = coll.separate_patches()


# Step 2 - KSVD with these patches
dict_size = 1100 #overcomplete
sparsity = 1
D_R, Gamma_R = k_svd(R, dict_size, sparsity)
D_G, Gamma_G = k_svd(G, dict_size, sparsity)
D_B, Gamma_B = k_svd(B, dict_size, sparsity)


end = time.time()
print("SUCCESS")
print("Time :", round(end-start,3), "seconds")

# Step 3 - Sort by decreasing variance the columns of D
#sort_patches ????

# Step 4 - Load one noisy image and patches
#choice = './Photos/2007061708_cam01.jpg'
#noisy_image = cv2.imread(choice)
#noisy_coll = Patch_collection(noisy_image, nb=1000, size_patch=32)

# Step 5 - Grid Patches
#noisy_coll.grid_patches()
#Rnoisy, Gnoisy, Bnoisy = noisy_coll.separate_patches()

# Step 6 - Mean and std each patch and new xbi
# ???? 

# Step 7 - OMP(xbi, D) = alphabi
#alphaR = omp(D, Rnoisy, sparsity)
#alphaG = omp(D, Gnoisy, sparsity)
#alphaB = omp(D, Bnoisy, sparsity)

# Step 8 - Dalphabi = Xfin
# X_R = np.mult(D, alphaR)
# X_G = np.mult(D, alphaG)
# X_B = np.mult(D, alphaB)

# Step 9 - Reconstruct
#noisy_coll.assemble_patches(X_R, X_G, X_B) ## à faire ????
#reconstruct_image = noisy_coll.reconstruct_image()
#cv2.imshow('image reconstructed', reconstruct_image)
#cv2.destroyAllWindows()