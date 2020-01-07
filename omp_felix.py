import numpy as np
from np.linalg import norm, pinv
from np import matmul, transpose
iter_max = 200


def omp(x, D, eps):
	k = D.shape[1]
	R = x
	alpha = np.zeros((k, 1))
	phi = []
	P = []
	n = 0
	while norm(R) > eps & & n < iter_max:
		n += 1
		tmp = np.abs(matmul(transpose(D), R) / norm(D, axis=1))
		p_k = np.argmax(tmp)
		phi = phi.append(D[:, p_k])
		P = P.append(p_k)

		phi_t = transpose(phi)
		intermediaire = pinv(matmul(phi_t, phi))
		intermediaire = matmul(intermediaire, phi_t)
		intermediaire = matmul(intermediaire, x)
		alpha[P] = intermediaire

		intermediaire = matmul(phi, pinv(matmul(phi_t, phi)))
		intermediaire = matmul(intermediaire, phi_t)
		intermediaire = matmul(intermediaire, x)
		new_R = x - intermediaire

		if norm(new_R) > norm(R):
			break
		R = new_R
	return alpha
