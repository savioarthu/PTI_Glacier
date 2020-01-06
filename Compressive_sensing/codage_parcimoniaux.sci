// Algorithme du Matching Pursuit
function alpha = MP(D,X,kmax,Epsilon)
    // D le dictionnaire
    // X un vecteur
    // kmax le nbr d'itérations maximum
    // Epsilon une valeur d'arrêt
    
    n=size(D,"c") // n représente le nombre de colonnes de D
    
    //Initialisation 
    alpha=zeros(n,1) // Alpha est initialisée comme une matrice de zéros
    residuel=X // A l'étape 0, le residuel vaut x.
    stop=norm(residuel) // Notre critère d'arrêt est la norme de notre résiduel
    iter=0 // Nombre d'itérations réalisées initialisé à 0
    
    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((iter<kmax) & (stop>Epsilon))
        // Sélection de l'atome qui contribue le plus au résiduel R^(0)
        for i = 1:n
            liste(i)=abs(D(:,i)'*residuel)/norm(D(:,i))
        end
        // Ainsi on retient l'indice correspondant au maximum 
        [_,mk]=max(liste)
        
        // On met à jour les coefficients de notre représentation parcimonieuse 
        alpha(mk)=alpha(mk)+(D(:,mk)'*residuel)/norm(D(:,mk))^2
        // On met à jour notre résiduel
        residuel=residuel-(residuel'*D(:,mk)/norm(D(:,mk))^2)*D(:,mk)
        // On met à jour notre critère d'arrêt
        stop=norm(residuel) 
        // On met à jour notre nombre d'itérations et on recommencer la boucle
        iter=iter+1
    end
endfunction

// Algorithme de l'OMP
function Alpha = OMP(D,X,kmax,Epsilon)
    // D le dictionnaire
    // X  un vecteur
    // kmax le nbr d'itérations maximum
    // Epsilon une valeur d'arrêt
    
    n=size(D,"c") // n représente le nombre de colonnes de D
    
    // Initialisation
    Alpha=zeros(n,1) // Alpha est initialisée comme une matrice de zéros
    residuel=X // A l'étape 0, le residuel vaut x.
    phi=[] // Phi est vide à l'étape 0. Il représente le dictionnaire actif.
    stop=norm(residuel) // Notre critère d'arrêt est la norme de notre résiduel
    iter=0 // Nombre d'itérations réalisées initialisé à 0
    P=[] // P représente l'ensemble des indices des coefficients. Il est vide à l'étape 0.
    C = zeros(n,1); // Vecteur des valeurs max de chaque colonne de D
    
    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((iter<kmax) & (stop>Epsilon))
        // Sélection de l'atome identique au MP : celui qui contribue le plus au résiduel R^(0)
        for i = 1:n
            if (norm(D(:,i))~=0)
                C(i)=abs(D(:,i)'*residuel)/norm(D(:,i))
            end
        end
        // Ainsi on retient toujours l'indice correspondant au maximum 
        [_,mk]=max(C)
        // disp(mk) // Pour vérifier l'atome sélectionné
        
        // On met à jour l'ensemble des indices P
        P=[P mk]
        // Construction de la matrice phi des colonnes Dmk (le dictionnaire actif)   
        phi=[phi D(:,mk)]
        
        // On met à jour les coefficients de notre représentation parcimonieuse 
        Alpha(P)=pinv(phi'*phi)*phi'*X
        // On met à jour notre résiduel        
        residuel=X-phi*pinv(phi'*phi)*phi'*X
        // On met à jour notre critère d'arrêt
        stop=norm(residuel)
        // On met à jour notre nombre d'itérations et on recommencer la boucle
        iter=iter+1
    end
endfunction

// Algorithme du KSVD
function [D,Gamma]=KSVD(X,D,Gamma,k)
    // D le dictionnaire
    // X un vecteur
    // Gamma la matrice telle que X=D*Gamma
    // k le nombre d'atomes souhaités dans le dictionnaire
    
    // Initialisation - Etape 0
    // Le dictionnaire est en paramètre donc déjà initialisé
    compteur=1 // Compteur nécessaire pour connaître le nombre de zéro 
    Mat=zeros(size(D,"r"),size(Gamma,"c"))
    
    // Etape 1 à k
    for i=1:k
        
        // On commence par calculer l'erreur Err sur les l signaux sans tenir compte de la contribution de la ième colonne de D
        for j=1:k
            if (i~=j)
                Mat=Mat+D(:,j)*Gamma(j,:)
            end
        end
        Err=X-Mat
        
        // On ne garde que les coefficients non nuls de Gamma qu'on stocke dans wi le support, c'est à dire le vecteur des positions des coefficients non nuls.
        G=Gamma(i,:)
        for j=1:size(Gamma,"c")
            if (G(j)~=0)
                wi(compteur)=j
                compteur=compteur+1
            end
        end
        compteur=1
        
        // Si ce support est vide, cela ne sert à rien de continuer et on peut passer à l'atome suivant.
        if (length(wi)==0)
            break
        end
        
        // Représentation de Oméga composée uniquement de 0 ou de 1 permettant d'exprimer l'erreur de reconstruction par la suite.
        OMEGA=zeros(size(X,"c"),length(wi))
        for j=1:length(wi)
            OMEGA(wi(j),j)=1     
        end
            
        // Erreur de reconstruction sans tenir compte des atomes correspondant aux coefficients non nuls de Gamma
        ERR=Err*OMEGA
            
        // On réalise enfin une décomposition SVD de ERR
        [U,S,V]=svd(ERR)
            
        // Mise à jour du dictionnaire D 
        D(:,i)=U(:,1)
            
        // Mise à jour de Gamma
        Gamma(i,wi) = S(1,1)*V(1,:)
        //disp(Gamma, "Gamma =")   
    end
endfunction


// Algorithme d'apprentissage d'un dictionnaire par KSVD
function [D,Gamma]=Apprentissage(X,k,L,Epsilon,kmax)
    // X une matrice de vecteurs d'apprentissage 
    // k le nombre d'atomes souhaités dans le dictionnaire
    // L le nombre de mises à jour 
    // Epsilon une valeur d'arrêt
    // kmax nombre d'itérations maximum
    // Methode la méthode de codage parcimonieux utilisée
    
    // Initialisation
    D=X(:,1:k)
    // Normalisation des colonnes de D avec les k premières colonnes de X
    for j=1:k
            D(:,j)=(D(:,j)-mean(D(:,j)))/(norm(D(:,j))^2)
    end
    // Gamma de taille k,l
    Gamma=zeros(k,size(X,"c"))

    // On répète L fois les étapes suivantes
    for j=1:L
        for i=1:size(X,"c")
            // On met à jour les coefficients de Gamma
            Gamma(:,i)=OMP(X(:,i),D)
            //disp(i)
        end
         disp(Gamma)
         // Mise à jour du dictionnaire de de Gamma
        [D,Gamma] = atomeKSVD(D,X,Gamma)
        //disp(Gamma)
        //disp(D, "D=")
        //disp(j)
    end
endfunction



//Algorithme de STAGE ORTHOGONAL MATCHING PURSUIT
function Alpha = StOMP(D,X,kmax,Epsilon)
    // D le dictionnaire
    // X  un vecteur
    // kmax le nbr d'itérations maximum
    // Epsilon une valeur d'arrêt
    
    n=size(D,"c") // n représente le nombre de colonnes de D
    
    // Initialisation
    Alpha=zeros(n,1) // Alpha est initialisée comme une matrice de zéros
    residuel=X // A l'étape 0, le residuel vaut x.
    phi=[] // Phi est vide à l'étape 0. Il représente le dictionnaire actif.
    stop=norm(residuel) // Notre critère d'arrêt est la norme de notre résiduel
    iter=0 // Nombre d'itérations réalisées initialisé à 0
    P=[] // P représente l'ensemble des indices des coefficients. Il est vide à l'étape 0.
    W=[]
    
    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((iter<kmax) & (stop>Epsilon))
        
        // Calcul du seuillage S
        t=2.3// t doit être compris entre 2 et 3
        Sk=(t*norm(residuel))/sqrt(n)

        // Calcul de la contribution de tous les atomes
        for i = 1:n
            // Sélection
            // On retient les indices >Sk le seuillage calculé précèdemment
            if (abs(D(:,i)'*residuel)/norm(D(:,i))>Sk)
                W=[W i]
            end
        end
        
        // Construction de la matrice phi des colonnes DI (le dictionnaire actif)   
        phi=[phi D(:,W)]
        
        // On met à jour l'ensemble des indices P
        P=[P W]

        // On met à jour les coefficients de notre représentation parcimonieuse 
        Alpha(P)=pinv(phi'*phi)*phi'*X
        // On met à jour notre résiduel        
        residuel=X-phi*pinv(phi'*phi)*phi'*X
        // On met à jour notre critère d'arrêt
        stop=norm(residuel)
        // On remet à jour l'ensemble des indices
        compteur=1
        for j=1:n
            if Alpha(j)~=0
                P(compteur)=j
            end
        end
        // On met à jour notre nombre d'itérations et on recommencer la boucle
        iter=iter+1
    end
endfunction



function res = PSNR(S, k)
   k = min(k + 1, min(size(S)));
   res = 10 * log10(255 ** 2 / S(k, k));
endfunction

