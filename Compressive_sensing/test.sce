//// Taille du vecteur d'origine
//N=800
//
////Algorithme de l'Orthogonal Matching Pursuit - OMP
//function alpha = OMP(X, D)
//    // D le dictionnaire
//    // X un vecteur
//    
//    K = size(D,2) // K représente le nombre de colonnes de D
//    
//    //Initialisation
//    Epsilon = 10^(-6); // La précision souhaitée servant de valeur d'arrêt
//    iter = 0; // Nombre d'itérations réalisées initialisé à 0
//    residuel = X; //init du résiduel 0 égal au signal
//    kmax = 200 //K/10; // Le nombre d'itéaration maximum
//    alpha = zeros(K, 1) // Alpha est initialisée comme une matrice de zéros 
//    phi = [] // Phi est vide à l'étape 0. Il représente le dictionnaire actif.
//    C = zeros(K,1); // Vecteur des valeurs max de chaque colonne de D
//    P = [] // P représente l'ensemble des indices des coefficients. Il est vide à l'étape 0.
//
//
//    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
//    while ((kmax > iter) & (norm(residuel) > Epsilon))
//        // Sélection de l'atome identique au MP : celui qui contribue le plus au résiduel R^(0
//        for i = 1:K
//            if (norm(D(:,i))<>0)
//                C(i) = abs(D(:,i)' * residuel) / norm(D(:,i));
//             end
//        end
//        // Ainsi on retient toujours l'indice correspondant au maximum 
//        [_,mk] = max(C);
//        // disp(mk) // Pour vérifier l'atome sélectionné
//        
//        // On met à jour l'ensemble des indices P
//        P = [P mk];
//        // Construction de la matrice phi des colonnes Dmk (le dictionnaire actif) 
//        phi = [phi D(:, mk)]
//
//        // On met à jour les coefficients de notre représentation parcimonieuse
//        alpha(P) = pinv(phi' * phi) * phi' * X;
//        
//        // On met à jour notre résiduel
//        residuel = X - phi * pinv(phi' * phi) * phi' * X;
//        // On met à jour notre nombre d'itérations et on recommencer la boucle
//        iter = iter + 1;
//    end
//endfunction
//
//
////Algorithme de STAGE ORTHOGONAL MATCHING PURSUIT - StOMP
//function alpha=StOMP(X,D,t)
//    // D le dictionnaire
//    // X  un vecteur
//    // t doit être compris entre 2 et 3
//    
//    K = size(D,2); // K représente le nombre de colonnes de D
//    
//    //Initialisation
//    Epsilon = 10^(-6); // La précision souhaitée servant de valeur d'arrêt
//    iter = 0; // Nombre d'itérations réalisées initialisé à 0
//    residuel = X; //init du résiduel 0 égal au signal
//    kmax = K/10; // Le nombre d'itéaration maximum
//    alpha = zeros(K, 1); // Alpha est initialisée comme une matrice de zéros 
//    phi = []; // Phi est vide à l'étape 0. Il représente le dictionnaire actif.
//    C = zeros(K,1); // Vecteur des valeurs max de chaque colonne de D
//    P = []; // P représente l'ensemble des indices des coefficients. Il est vide à l'étape 0.
//    
//    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
//    while ((kmax > iter) & (norm(residuel) > Epsilon))
//        // Calcul de la contribution de tous les atomes
//        for i = 1:K
//            if (norm(D(:,i))<>0)
//                C(i) = abs(D(:,i)' * residuel) / norm(D(:,i));
//             end
//        end
//        
//        // Calcul du seuillage Sk
//        Sk=t*norm(residuel)/sqrt(K);
//        
//        // On retient les indices >Sk le seuillage calculé précèdemment
//        W=find(C(i)>=Sk)
//        
//        // On met à jour l'ensemble des indices P
//        P = [P W]
//        
//        // Construction de la matrice phi des colonnes DI (le dictionnaire actif)  
//        for  i=1:length(W)
//            phi=[phi D(:,W(i))];
//        end
//
//        // On met à jour les coefficients de notre représentation parcimonieuse
//        alpha(P) = pinv(phi' * phi) * phi' * X;
//        
//        // On met à jour notre résiduel 
//        residuel = X - phi * pinv(phi' * phi) * phi' * X;
//        
//        // On remet à jour l'ensemble des indices
//        P=find(alpha<>0);
//        
//        // On met à jour notre nombre d'itérations et on recommencer la boucle
//        iter = iter + 1;
//    end
//endfunction

// Matrice aléatoire générée à partir d'un processus uniformément distribuée
function phi=phi1(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100);
    phi=rand(M,N);
endfunction

// Matrice aléatoire générée à partir d'un processus bernoullien {-1,1}
function phi=phi2(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100);
    p=0.5;
    phi=grand(M,N,"bin",1,p)*2-1;
endfunction

// Matrice aléatoire générée à partir d'un procesus bernoullien {0,1}
function phi=phi3(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100);
    p=0.5;
    phi=grand(M,N,"bin",1,p);
endfunction

// Matrice aléatoire générée à partir d'un processus gaussien identique et idépendamment distribué (i.i.d) avec une moyenne nulle et une variance 1/M : N(0,1//M)
function phi=phi4(P,N)
     // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100);
    phi=grand(M,N,"nor",0,sqrt(1/M));
endfunction

// Matrice creuse ou parcimonieuse générée de façon indépendante 
function phi=phi5(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
     
    // M le nombre de mesures considérées
    M=floor(P*N/100);
    p=0.5;
    phi=full(sprand(M,N,p));
endfunction

// Algorithme de la cohérence mutuelle
function C=coherence_mutuelle(phi,N,D)
    // phi : matrice de mesure
    // N : taille du vecteur d'origine
    // D : dictionnaire d'apprentissage
    
    // Initialisation du maximum
    maxi=0;
    
    // La cohérence mutuelle est égale à la plus grande valeur absolue du produit scalaire entre les vecteurs colonnes de D et les vecteurs lignes de phi
    for i=1:size(phi,"r")
        for j=1:size(D,"c")
            if i<>j
                if maxi< abs(phi(i,:)*D(:,j))/(norm(phi(i,:))*norm(D(:,j)))
                    maxi=max(abs(phi(i,:)*D(:,j))/(norm(phi(i,:))*norm(D(:,j))));
                end
            end
        end
    end
    
    C=sqrt(N)*maxi;
    // La cohérénce mutuelle doit être entre  1 et sqrt(N)
endfunction

// Calcul de la cohérence d'une matrice
function C=coherence(A)
    // A : A=Phi*D une matrice
    
    // Initialisation du maximum
    maxi = 0;
    
    // La cohérence de A est égale à la plus grande valeur absolue du produit scalaire entre deux vecteurs colonnes distincts de A
    for i = 1:size(A,"r")
        for j =1:size(A,"r")
            if i<>j
                if maxi< abs(A(:,i)'*A(:,j))/(norm(A(:,i))*norm(A(:,j)))
                    maxi=max(abs(A(:,i)'*A(:,j))/(norm(A(:,i))*norm(A(:,j))));
                end
            end
        end
    end
    C=maxi;
endfunction

// Calcul des vecteurs de mesure Yi
function Z=vecteurs_mesures(phi,X)
    // Phi : matrice de mesure
    // X : Matrice du signal d'origine
    
    // On calcule le produit phi*X pour chaque colonne de X
    l=size(X,"c");
    Z=[];
    for i = 1:l
        Z(:,i)=phi*X(:,i);
    end
endfunction

// Calcul de l'erreur quadratique entre deux matrices
function M=MSE(A,B)
    // A : premier élement à comparer
    // B : second élément à comparer
    
    // Calcul de l'erreur quadratique moyenne entre A et B
    n=size(A,"r");
    for i=1:n
        M=(A(i)-B(i))^2;
    end
    M=(1/n)*M;
endfunction


// Execution du script des fonctions

//// Path à modifier si emplacement du fichier different
//load('dico.dat') // Renvoie un dictionnaire de taille 800 x 800
//load('signalOriginal.dat') // Renvoie une matrice X de taille 800 x 1
//
//// On prend tous l'espace nécessaire possible pour l'exécution du programme
//stacksize('max')
//
//// Verification parcimonieux
//X = signalOriginal;
//D = dico ;
//alp = OMP(X,D);
//
//// Etape 1
//// a)
//
//// Pourcentage de mesure P variant entre 10 et 100
//P=[10,15,25,30,35,40,50,75,100]
//   
//// Initialisation des vecteurs contenant les coherences mutuelles en fonction de P   
//C_phi1=[]
//C_phi2=[]
//C_phi3=[]
//C_phi4=[]
//C_phi5=[]
//
//// Calcul de la cohérence mutuelle de chaque phi pour chaque P 
//for i = 1:length(P)
//    P1=phi1(P(i),N);
//    C1=coherence_mutuelle(P1,N,D);
//    C_phi1(i)=C1;
//    
//    P2=phi2(P(i),N);
//    C2=coherence_mutuelle(P2,N,D);
//    C_phi2(i)=C2;
//    
//    P3=phi3(P(i),N);
//    C3=coherence_mutuelle(P3,N,D);
//    C_phi3(i)=C3;
//    
//    P4=phi4(P(i),N);
//    C4=coherence_mutuelle(P4,N,D);
//    C_phi4(i)=C4;
//    
//    P5=phi5(P(i),N);
//    C5=coherence_mutuelle(P5,N,D);
//    C_phi5(i)=C5;
//end
//
//// Affichage de la cohérence mutuelle pour chaque P et chaque phi
//disp("Cohérence mutuelle")
//
////disp(P, "P = ")
////
////disp(C_phi1', "Pour phi1, C=")
////disp(C_phi2', "Pour phi2, C=")
////disp(C_phi3', "Pour phi3, C=")
////disp(C_phi4', "Pour phi4, C=")
////disp(C_phi5', "Pour phi5, C=")
//
//// Illustration graphique de l'évolution de la cohérence mutuelle en fonction de P
//plot2d(P, [C_phi1 C_phi2 C_phi3 C_phi4 C_phi5], rect=[10,1,100,sqrt(N)])
//legends(["Phi1"; "Phi2" ; "Phi3" ; "Phi4" ; "Phi5"],[1,2,3,4,5],4)
//xtitle('Evolution de la cohérence mutuelle en fonction de P')
//
//
//// b)
//// Initialisation des vecteurs 
//C_phi1=[]
//C_phi2=[]
//C_phi3=[]
//C_phi4=[]
//C_phi5=[]
//
//// Calcul pour chaque phi pour chaque P 
//for i = 1:length(P)
//    P1=phi1(P(i),N);
//    // Calcul des vecteurs de mesure pour phi1 pour chaque P
//    Y1=vecteurs_mesures(P1,X);
//    // Reconstruction du signal
//    A1=P1*D;
//    alpha1=OMP(Y1,A1);
//    // Approximation du signal d'origine
//    X1=D*alpha1;
//    // Calcul d'erreur
//    M1=MSE(X1,X);
//    // Stockage
//    C_phi1(i)=M1;
//    
//    P2=phi2(P(i),N);
//    // Calcul des vecteurs de mesure pour phi2 pour chaque P
//    Y2=vecteurs_mesures(P2,X);
//    // Reconstruction du signal
//    A2=P2*D;
//    alpha2=OMP(Y2,A2);
//    // Approximation du signal d'origine
//    X2=D*alpha2;
//    // Calcul d'erreur
//    M2=MSE(X2,X);
//    // Stockage
//    C_phi2(i)=M2;
//    
//    P3=phi3(P(i),N);
//    // Calcul des vecteurs de mesure pour phi3 pour chaque P
//    Y3=vecteurs_mesures(P3,X);
//    // Reconstruction du signal
//    A3=P3*D;
//    alpha3=OMP(Y3,A3);
//    // Approximation du signal d'origine
//    X3=D*alpha3;
//    // Calcul d'erreur
//    M3=MSE(X3,X);
//    // Stockage
//    C_phi3(i)=M3;
//    
//    P4=phi4(P(i),N);
//    // Calcul des vecteurs de mesure pour phi4 pour chaque P
//    Y4=vecteurs_mesures(P4,X);
//    // Reconstruction du signal
//    A4=P4*D;
//    alpha4=OMP(Y4,A4);
//    // Approximation du signal d'origine
//    X4=D*alpha4;
//    // Calcul d'erreur
//    M4=MSE(X4,X);
//    // Stockage
//    C_phi4(i)=M4;
//    
//    P5=phi5(P(i),N);
//    // Calcul des vecteurs de mesure pour phi5 pour chaque P
//    Y5=vecteurs_mesures(P5,X);
//    // Reconstruction du signal
//    A5=P5*D;
//    alpha5=OMP(Y5,A5);
//    // Approximation du signal d'origine
//    X5=D*alpha5;
//    // Calcul d'erreur
//    M5=MSE(X5,X);
//    // Stockage
//    C_phi5(i)=M5;
//end
//
//
//// Affichage de l'erreur pour chaque P et chaque phi
//disp("Erreur quadratique moyenne")
//
////disp(P, "P = ")
////
////disp(C_phi1', "Pour phi1, E=")
////disp(C_phi2', "Pour phi2, E=")
////disp(C_phi3', "Pour phi3, E=")
////disp(C_phi4', "Pour phi4, E=")
////disp(C_phi5', "Pour phi5, E=")
//
//// Illustration graphique de l'évolution de l'erreur en fonction de P pour chaque phi
//plot2d(P, [C_phi1 C_phi2 C_phi3 C_phi4 C_phi5]) //, rect=[10,1,100,sqrt(N)])
//legends(["Phi1"; "Phi2" ; "Phi3" ; "Phi4" ; "Phi5"],[1,2,3,4,5],4)
//xtitle("Evolution de lerreur en fonction de P pour chaque phi")
//
//


////subplot(2,3,1)
//id=winsid()
//old=id($)
//new=old+1
//f=scf(new)
//plot2d(X,rect=[0,-2,800,2])
//id=winsid()
//old=id($)
//new=old+1
//f=scf(new)
////subplot(2,3,2)
//plot(X1)
//id=winsid()
//old=id($)
//new=old+1
//f=scf(new)
////subplot(2,3,3)
//plot(X2)
//id=winsid()
//old=id($)
//new=old+1
//f=scf(new)
////subplot(2,3,4)
//plot(X3)
//id=winsid()
//old=id($)
//new=old+1
//f=scf(new)
////subplot(2,3,5)
//plot(X4)
//id=winsid()
//old=id($)
//new=old+1
//f=scf(new)
////subplot(2,3,6)
//plot(X5)
