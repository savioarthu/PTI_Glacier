//Algo de l'Orthogonal Matching Pursuit (OMP)
function [alpha] = OMP(x, D) //x le vecteur signal, D la matrice du dictionnaire
    //Init var
    nbIterMax = 10; //max d'iterations souhaitées 
    eps = 10^(-6); //précision souhaitée
    nbIter = 0; //compteur d'itérations
    R = x; //init du résiduel 0 égal au signal
    K = size(D,2) //on récup K = nombre de colonnes du dictionnaire D
    alpha = zeros(K, 1) //init du vecteur des contributions à K lignes 
    phi = [] //init de la matrice du dictionnaire 
    PS = zeros(K,1); //init du vecteur des valeurs max de chaque colonne de D à K lignes
    Pk = [] //init de la matrice qui met à jour l'ensemble des indices


    //Conditions d'arrêt 
    while ((nbIterMax > nbIter) & (norm(R) > eps))
        //On cherche l'atome qui contribue le plus au résiduel de l'étape précédente 
        //On calcule la valeur max pour chaque colonne de D (pour rechercher l'atome à la plus petite erreur ie celui qui
        //contribue le plus au résiduel précédent)
        //disp(size(D))
        for i = 1:K
            if (norm(D(:,i))<>0)
                PS(i) = abs(D(:,i)' * R) / norm(D(:,i));
             end
        end
        //On cherche l'indice de l'atome qui contribue le plus au résiduel de l'étape précédente (mk correspond à l'indice)
        //C'est donc le max du max de toutes les colonnes de D, ie le max de PS
        [Max ,mk] = max(PS);
        //On rajoute l'indice mk à la matrice qui retient et met à jour les indices des étapes précédentes et à l'étape
        //courante
        Pk = [Pk mk];
        //On met à jour la matrice du dictionnaire actif en ajoutant l'atome trouvé
        phi = [phi D(:, mk)]

        //On met à jour la représentation des atomes aux étapes précédentes et à l'étape courante
        alpha(Pk) = pinv(phi' * phi) * phi' * x;
        
        //On calcule le nouveau résiduel 
        R = x - phi * pinv(phi' * phi) * phi' * x;
        nbIter = nbIter + 1;
        //disp(size(alpha), 'size alpha')
    end
endfunction

 
