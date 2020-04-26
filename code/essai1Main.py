#!/usr/bin/python3
# 2020 - IMSI - Projet Traitement des images
# Arthur Savio - Félix Cloup - Yoann Ruiz - Maxime Lesaffre - Alexis Teste
# Fonction principale

# Import des fonctions
from essai1Functions import *

start = time()

# Lecture des images
paths = ['./../Photos/2007060208_cam01.jpg', './../Photos/2007060608_cam01.jpg',
         './../Photos/2007060908_cam01.jpg', './../Photos/2007061708_cam01.jpg']
paths_chat = ['./../Photos_test/chat1.jpg', './../Photos_test/chat2.jpg']

images = [cv2.imread(path) for path in paths_chat]


# Extraction de patchs
coll_learn = Patch_collection(images[0], nb=100, size_patch=32)
coll_denoise = Patch_collection(images[0], nb=100, size_patch=32)

# Ksvd 
ksvd = KSVD(collection_aprentissage=coll_learn, collection_debruitage=coll_denoise)

# Débruitage
ksvd.denoise(show_output=True, save_output=True)



end = time()
print("Temps execution :", round(end-start, 3), "secondes")