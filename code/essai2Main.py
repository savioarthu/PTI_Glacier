#!/usr/bin/python3
# 2020 - IMSI - Projet Traitement des images
# Arthur Savio - Félix Cloup - Yoann Ruiz - Maxime Lesaffre - Alexis Teste
# Fonction principale

# Import des fonctions
from essai2Functions import *


start = time()


# Paramètres

sigma = 20 
num_dict = 300



# Lecture des images

learning_image = cv2.imread('./../Photos_test/glacier1-carre.jpg', 0)
noisy_image  = cv2.imread('./../Photos_test/noisy_glacier-carre.jpg', 0)



# Bruit sur image si nécessaire

#noisy_image = #cv2.imread(paths_chat[1])
#noise_layer = np.random.normal(0, sigma ^ 2, image.size).reshape(image.shape).astype(int)
#noisy_image = image # noise_layer



# Resize image si nécessaire

max_init_size = max(learning_image.shape[0], learning_image.shape[1])
resize_ratio = max_resize_dim/max_init_size
learning_image = learning_image * 1.0

if resize_ratio < 1:
    learning_image = cv2.resize(learning_image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)


# Débruitage image

denoised_image = denoising(noisy_image, learning_image, dict_size=num_dict, sparsity=max_sparsity)


# Calcul PSNR

psnr = psnr(noisy_image, denoised_image)
print(psnr)


# Affichage images

#numpy_horizontal_concat = np.concatenate((image.astype('uint8'), noisy_image.astype('uint8')), axis=1)
#numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, denoised_image.astype('uint8')), axis=1)
numpy_horizontal_concat = np.concatenate((noisy_image.astype('uint8'), denoised_image.astype('uint8')), axis=1)
cv2.namedWindow('Original vs Reconstruit', cv2.WINDOW_NORMAL)
imS = cv2.resize(numpy_horizontal_concat, (1500, 1040))
cv2.imshow("Original vs Reconstruit", imS)
cv2.waitKey(0)

end = time()
print("Temps exécution :", round(end-start, 3), "secondes")



