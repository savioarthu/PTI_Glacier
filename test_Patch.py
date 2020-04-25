from Patch_collection import *
from apprentissage import *
import cv2
import time

start = time.time()
paths = ['./Photos/2007060208_cam01.jpg', './Photos/2007060608_cam01.jpg',
         './Photos/2007060908_cam01.jpg', './Photos/2007061708_cam01.jpg']
paths_chat = ['./Photos_test/chat1.jpg', './Photos_test/chat2.jpg']


images = [cv2.imread(path) for path in paths_chat]

coll = Patch_collection(images, nb=1024, size_patch=32)

""" coll = Patch_collection(images, nb=3, size_patch=32)

coll.select_patches(nb_iter=3, threshold=0.01, fill=False)

coll.patches[0].show() """

# print(0)
# coll.grid_patches()
# print(1)
#image = coll.reconstruct_image()
# print(2)
#cv2.imshow('image reconstructed', image)

# for patch in coll.patches:
# 	patch.show(corrected=True)

# while True:
#	if cv2.waitKey(1) & 0xFF == ord('q'):
#		break

# cv2.destroyAllWindows()


# Step 1 - Recover patches
coll.select_patches(nb_iter=3, threshold=0.01, fill=False)
R, G, B = coll.separate_patches()

# Step 2 - KSVD with these patches
dict_size = 1100  # overcomplete
sparsity = 1
D_R, Gamma_R = k_svd(R, dict_size, sparsity)
D_G, Gamma_G = k_svd(G, dict_size, sparsity)
D_B, Gamma_B = k_svd(B, dict_size, sparsity)


end = time.time()
print("SUCCESS")
print("Time :", round(end-start, 3), "seconds")

# Step 3 - Sort by decreasing variance the columns of D
variance = np.var(D_R, axis=1)
sorted_result = np.argsort(-1*variance)
D_R = D_R[sorted_result]
variance = np.var(D_G, axis=1)
sorted_result = np.argsort(-1*variance)
D_G = D_G[sorted_result]
variance = np.var(D_B, axis=1)
sorted_result = np.argsort(-1*variance)
D_B = D_B[sorted_result]

# Step 4 - Load one noisy image and patches
choice = './Photos/2007061708_cam01.jpg'
noisy_image = cv2.imread(choice)
noisy_coll = Patch_collection(noisy_image, nb=1024, size_patch=32)

# Step 5 - Grid Patches
noisy_coll.grid_patches()
Rnoisy, Gnoisy, Bnoisy = noisy_coll.separate_patches()

# Step 6 - Mean and std each patch and new xbi
# ????

X_R = np.zeros((Rnoisy.shape[0], Rnoisy.shape[1]))
X_G = np.zeros((Gnoisy.shape[0], Gnoisy.shape[1]))
X_B = np.zeros((Bnoisy.shape[0], Bnoisy.shape[1]))
# Step 7 - OMP(xbi, D) = alphabi
for i in range(0, len(Rnoisy)):
    alphaR = omp(D_R, Rnoisy[i].reshape((1024, 1)), sparsity)
    alphaG = omp(D_G, Gnoisy[i].reshape((1024, 1)), sparsity)
    alphaB = omp(D_B, Bnoisy[i].reshape((1024, 1)), sparsity)

    # Step 8 - Dalphabi = Xfin
    X_R[i] = np.matmul(D_R, alphaR).reshape(1024)
    X_G[i] = np.matmul(D_G, alphaG).reshape(1024)
    X_B[i] = np.matmul(D_B, alphaB).reshape(1024)

# Step 9 - Reconstruct
# X_R X_G X_B de taille 17424 / 1024
# Il faut reconstruire en un seul X -> ce que je fais déjà reconstruct_image.
# Il faut donc juste rentrer X_R X_G et X_B dans la classe ou faire differemment de ce que j'ia fait
reconstruct_image = noisy_coll.reconstruct_image()
cv2.imshow('image reconstructed', reconstruct_image)
cv2.destroyAllWindows()
