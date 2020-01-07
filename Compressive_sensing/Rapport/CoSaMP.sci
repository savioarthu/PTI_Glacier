//Algorithme du Compressive Sampling du Matching Pursuit - CoSaMP
function Alpha = CoSaMP(X, D, s)
    // D le dictionnaire
    // X un vecteur
    // S ordre de parcimonie
    
    K = size(D,2); // K représente le nombre de colonnes de D
    
    //Initialisation
    Epsilon = 10^(-6); // La précision souhaitée servant de valeur d'arrêt
    iter = 0; // Nombre d'itérations réalisées initialisé à 0
    residuel = X; //init du résiduel 0 égal au signal
    kmax = 200 //K/10; // Le nombre d'itéaration maximum
    Alpha = zeros(K, 1); // Alpha est initialisée comme un vecteur de zéros 
    Supp = []; // Le support
    Supp1 = []; // Le support de sélection
    C = []; // Vecteur des contributions
    
    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((kmax > iter) & (norm(residuel) > Epsilon))
        //Selection
        for i = 1:K
            if (norm(D(:,i))<>0)
                C(i) = abs(D(:,i)' * residuel) / norm(D(:,i));
            end
        end
        
        // Ainsi on retient les 2*s atomes correspondant au maximum
        for i=1:2*s
            [_,mk] = max(C);
            Supp1(i) = mk;
            C(mk)=0;
        end
        // Mise à jour du support
        if size(Supp)==[0,0]
            Supp=Supp1
        else
            for i=1:size(Supp1,"c")
                for j=1:size(Supp,"c")
                    if Supp1(i)<>Supp(j)
                        Supp($+1)=Supp1(i)
                    end
                end
            end
        end

        AS=D(:,Supp)
        // Estimation
        Alpha(Supp)=pinv(AS' * AS) * AS' * X;

        // Rejet
        // On conserve les s plus grands coefficients Z de alpha
        Z=Alpha
        newalpha=zeros(K,1)
        for i=1:s
            [val,mk] = max(Z);
            newalpha(mk) = val;
            Z(mk)=0;
        end
            
        Alpha=newalpha;
         // On met à jour notre résiduel
        residuel = X - D * Alpha
        // On met à jour notre nombre d'itérations et on recommencer la boucle
        iter = iter + 1;
    end
endfunction

