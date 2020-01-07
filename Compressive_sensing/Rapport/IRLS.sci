// Algorithme IRLS (Iteratively reweighted least squares)

function alpha=IRLS(X,D,p)
    // X vecteur
    // D Dictionnaire
    // p p>0
    
    K = size(D,2); // K représente la taille du nbr de colonnes de D
    k = 0; // nbr itérations 
    kmax=200; // nbr itérations max
    Epsilon=0.1; // Coefficient de régularisation
    alpha=D'*pinv(D*D')*X; // Initialisation d'alpha 

    // Calcul des poids W
    W=(alpha.^2+Epsilon).^((p/2)-1);
    // Boucle générale
    for i=1:K
        oldalpha=alpha
        Q=diag(W)
        alpha=Q*D'*pinv(D*Q*D')*X
        if ((abs(norm(alpha)-norm(oldalpha))>(sqrt(Epsilon)/100)) & (k<kmax))
            W=(alpha.^2+Epsilon).^((p/2)-1);
            k=k+1;
            break;
        elseif ((abs(norm(alpha)-norm(oldalpha))<(sqrt(Epsilon)/100)) & (Epsilon<10^-8))
            Epsilon=Epsilon/10;
            if k<kmax
                W=(alpha.^2+Epsilon).^((p/2)-1);
                k=k+1;
                break;
            else 
                break;
            end
        else
            break;
        end
    end
endfunction


