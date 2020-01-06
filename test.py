import numpy as np

# Algorithme KSVD 
def KSVD(D,X,Gamma):
	# D le dictionnaire
	# X le vecteur du signal d'origine
	# Gamma la matrice telle que X=D*Gamma

	K = len(D[1]) #K le nombre d'atomes souhaités dans le dictionnaire

	# Etape 1 à K
	for i in range(1, K):
		# On commence par calculer l'erreur Err sur les l signaux sans tenir compte de la contribution de la ième colonne de D
		Mat = []
		for j in range (1, K):
			if j != i:
				Mat.append(D[:,j] * Gamma[j,:])
				
		Err = X - Mat
		
		# On ne garde que les coefficients non nuls de Gamma qu'on stocke dans wi le support, c'est à dire le vecteur des positions des coefficients non nuls.
		wi = []
		for i in range(0, len(Gamma[0])):
			if Gamma[i,:] != 0:
				wi.append(i)
				
		# Si ce support est vide, cela ne sert à rien de continuer et on peut passer à l'atome suivant.
		if len(wi) == 0:
			break

		# Représentation de Oméga composée uniquement de 0 ou de 1 permettant d'exprimer l'erreur de reconstruction par la suite.
		OMEGA = np.zeros(len(X[1]),len(wi))
		for w in range(1, len(wi)):
			OMEGA[wi[w], w] = 1
		
		# Erreur de reconstruction sans tenir compte des atomes correspondant aux coefficients non nuls de Gamma
		ERR = Err * OMEGA

		# On réalise enfin une décomposition SVD de ERR
		U, S, V = np.linalg.svd(ERR)

		# Mise à jour du dictionnaire D
		D[:,i] = U[:,1]
		
		# Mise à jour de Gamma
		Gamma[i,wi] = V[1,:] * S[1,1]

	return D, Gamma
