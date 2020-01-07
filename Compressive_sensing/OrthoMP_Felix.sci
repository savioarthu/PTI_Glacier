iter_Max = 200

function alpha = OrthoMP(x, D, eps)
	[_, k] = size(D)
	R = x
	alpha = zeros(k, 1)
	phi = []
	m = []
	n = 0
	while norm(R) > eps & n < iter_Max then
		n = n + 1
		for i = 1:k
			tmp(i) = abs(D(:, i)' * R) / norm(D(:, i)) // Contribution de chaque atome sur le résiduel
		end
		[_, m_k] = max(tmp) // On prend l'atome qui à la plus grosse contribution
		phi = [phi D(:, m_k)]
		m = [m m_k]
		alpha(m) = pinv(phi' * phi) * phi' * x
		new_R = x - phi * pinv(phi' * phi) * phi' * x
		if norm(new_R) > norm(R) then
		// 	disp(new_R, "new_R");
		// 	disp(n, "n");
			break
		end
		R = new_R
	end
endfunction