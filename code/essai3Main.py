#!/usr/bin/python3
# 2020 - IMSI - Projet Traitement des images
# Arthur Savio - Félix Cloup - Yoann Ruiz - Maxime Lesaffre - Alexis Teste
# Fonction principale

from essai3Functions import *


# Paramètres

sigma = 10
window_shape = (16, 16) 
step = 4
ratio = 1.1
ksvd_iter = 5

# Chargement images

original_image = np.asarray(Image.open('./../Photos_test/noisy_glacier-min2.jpg').convert('L'))
learning_image = np.asarray(Image.open('./../Photos_test/noisy_glacier-min2.jpg').convert('L'))
noisy_image = original_image 


# Débruitage

start = time()
denoised_image = denoising(noisy_image, learning_image, window_shape, step, sigma, ratio, ksvd_iter)
psnr = psnr(noisy_image, denoised_image)
print('PSNR             : ' + str(psnr) + ' dB')

# Affichage

numpy_horizontal_concat = np.concatenate((noisy_image.astype('uint8'), denoised_image.astype('uint8')), axis=1)
cv2.imshow("Original vs Reconstruit", numpy_horizontal_concat)
cv2.waitKey(0)

end = time()
print("Temps exécution :", round(end-start, 3), "secondes")








