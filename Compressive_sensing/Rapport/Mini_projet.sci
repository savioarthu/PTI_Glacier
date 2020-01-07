// Mini projet script.sci
// Groupe Thibault LEDOUX - Arthur SAVIO - Pierre Soulier

clear
clc

// Données nécessaires 
// Path à modifier si emplacement du fichier different
load('data.dat')
// Renvoie une matrice XX de taille N*l avec N=99 et l =108

// Redefinition de fonctions
funcprot(0)

// On prend tous l'espace nécessaire possible pour l'exécution du programme
stacksize('max')

// Taille de X
X=XX; // Matrice composée des vecteurs d'apprentissage Xi
N=size(X,1); // N=99 - Lignes de X
l=size(X,2); // l=108 - Colonnes de X
K=100; // Taille souhaitée du dictionnaire
L=10; // Nombre d'itérations


// ------------------------------------------------------------------------


// Fonctions nécessaires préalables au mini projet


// Algorithme du Matching Pursuit   
function alpha = MP(X, D)
    // D le dictionnaire
    // X un vecteur
    
    n=size(D,"c") // n représente le nombre de colonnes de D
    
    //Initialisation 
    kmax=n/10; // nbr itérations maximum
    Epsilon=10^-6; // précision souhaitée 
    alpha=zeros(n,1) // Alpha est initialisée comme une matrice de zéros
    residuel=X // A l'étape 0, le residuel vaut x.
    stop=norm(residuel) // Notre critère d'arrêt est la norme de notre résiduel
    iter=0 // Nombre d'itérations réalisées initialisé à 0
    C=zeros(n,1); // Vecteur des contributions de chaqueatome de D
    
    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((iter<kmax) & (stop>Epsilon))
        // Sélection de l'atome qui contribue le plus au résiduel R^(0)
        for i = 1:K
            if (norm(D(:,i))<>0)
                C(i) = abs(D(:,i)' * residuel) / norm(D(:,i));
            end
        end
        // Ainsi on retient l'indice correspondant au maximum 
        [val,mk]=max(C)
        
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



//Algorithme de l'Orthogonal Matching Pursuit - OMP
function alpha = OMP(X, D)
    // D le dictionnaire
    // X un vecteur
    
    K = size(D,2) // K représente le nombre de colonnes de D
    
    //Initialisation
    Epsilon = 10^(-6); // La précision souhaitée servant de valeur d'arrêt
    iter = 0; // Nombre d'itérations réalisées initialisé à 0
    residuel = X; //init du résiduel 0 égal au signal
    kmax = 10; // Le nombre d'itéaration maximum
    alpha = zeros(K, 1) // Alpha est initialisée comme un vecteur de zéros 
    phi = [] // Phi est vide à l'étape 0. Il représente le dictionnaire actif.
    C = zeros(K,1); // Vecteur des contributions des atomes de D
    P = [] // P représente l'ensemble des indices des coefficients. Il est vide à l'étape 0.

    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((kmax > iter) & (norm(residuel) > Epsilon))
        // Sélection de l'atome identique au MP : celui qui contribue le plus au résiduel R^(0
        for i = 1:K
            if (norm(D(:,i))<>0)
                C(i) = abs(D(:,i)' * residuel) / norm(D(:,i));
            end
        end
        // Ainsi on retient toujours l'indice correspondant au maximum 
        [val,mk] = max(C);
        // disp(mk) // Pour vérifier l'atome sélectionné
        
        // On met à jour l'ensemble des indices P
        P = [P mk];
        // Construction de la matrice phi des colonnes Dmk (le dictionnaire actif) 
        phi = [phi D(:, mk)]

        // On met à jour les coefficients de notre représentation parcimonieuse
        alpha(P) = pinv(phi' * phi) * phi' * X;
        
        // On met à jour notre résiduel
        residuel = X - phi * pinv(phi' * phi) * phi' * X;
        // On met à jour notre nombre d'itérations et on recommencer la boucle
        iter = iter + 1;
    end
endfunction



//Algorithme de Stage Orthogonal Matching Pursuit - StOMP
function alpha=StOMP(X,D,t)
    // D le dictionnaire
    // X le vecteur du signal d'origine
    // t doit être compris entre 2 et 3
    
    K = size(D,2) // K représente le nombre de colonnes de D
    
    //Initialisation
    Epsilon = 10^(-6); // La précision souhaitée servant de valeur d'arrêt
    iter = 0; // Nombre d'itérations réalisées initialisé à 0
    residuel = X; //init du résiduel 0 égal au signal
    kmax = K/10; // Le nombre d'itérations maximum
    alpha = zeros(K, 1) // Alpha est initialisée comme une matrice de zéros 
    phi = [] // Phi est vide à l'étape 0. Il représente le dictionnaire actif.
    C = zeros(K,1); // Vecteur des valeurs max de chaque colonne de D
    P = [] // P représente l'ensemble des indices des coefficients. Il est vide à l'étape 0.
    
    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((kmax > iter) & (norm(residuel) > Epsilon))
        // Calcul de la contribution de tous les atomes
        for i = 1:K
            if (norm(D(:,i))<>0)
                C(i) = abs(D(:,i)' * residuel) / norm(D(:,i));
             end
        end
        
        // Calcul du seuillage Sk
        Sk=t*norm(residuel)/sqrt(K)
        
        // On retient les indices >Sk le seuillage calculé précèdemment
        W=find(C(i)>=Sk)
        
        // On met à jour l'ensemble des indices P
        P = [P W]
        
        // Construction de la matrice phi des colonnes DI (le dictionnaire actif)  
        for  i=1:length(W)
            phi=[phi D(:,W(i))]
        end

        // On met à jour les coefficients de notre représentation parcimonieuse
        alpha(P) = pinv(phi' * phi) * phi' * X;
        
        // On met à jour notre résiduel 
        residuel = X - phi * pinv(phi' * phi) * phi' * X;
        
        // On remet à jour l'ensemble des indices
        P=find(alpha<>0)
        
        // On met à jour notre nombre d'itérations et on recommencer la boucle
        iter = iter + 1;
    end
endfunction



//Algorithme KSVD 
function[D,Gamma] = KSVD(D,X,Gamma)
    // D le dictionnaire
    // X le vecteur du signal d'origine
    // Gamma la matrice telle que X=D*Gamma
    
    K = size(D,2); // K le nombre d'atomes souhaités dans le dictionnaire
    
    // Etape 1 à K
    for i=1:K
        // On commence par calculer l'erreur Err sur les l signaux sans tenir compte de la contribution de la ième colonne de D
        Mat=[]
        for (j = 1:size(D,2))
            if (j <> i) then
                Mat = Mat + (D(:,j) * Gamma(j,:));
            end
        end
        Err = X - Mat
        
        // On ne garde que les coefficients non nuls de Gamma qu'on stocke dans wi le support, c'est à dire le vecteur des positions des coefficients non nuls.
        wi = find(Gamma(i,:) <> 0);
        
        // Si ce support est vide, cela ne sert à rien de continuer et on peut passer à l'atome suivant.
        if (length(wi)==0)
            break
        end
        
        // Représentation de Oméga composée uniquement de 0 ou de 1 permettant d'exprimer l'erreur de reconstruction par la suite.
        OMEGA = zeros(size(X,2),length(wi));
        for w = 1:length(wi)
            OMEGA(wi(w),w) = 1;
        end

        // Erreur de reconstruction sans tenir compte des atomes correspondant aux coefficients non nuls de Gamma
        ERR = Err * OMEGA;

        // On réalise enfin une décomposition SVD de ERR
        [U,S,V] = svd(ERR);
        
        // Mise à jour du dictionnaire D
        D(:,i) = U(:,1);



        
        // Mise à jour de Gamma
        Gamma(i,wi) = V(1,:)*S(1,1);
    end 
endfunction


// Algorithme d'apprentissage d'un dictionnaire par KSVD pour OMP
function [D,Gamma]=Apprentissage_OMP(X,k,L)
    // X le vecteur du signal d'origine 
    // k le nombre d'atomes souhaités dans le dictionnaire
    // L le nombre de mises à jour 

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
            Gamma(:,i)=OMP(X(:,i),D);
        end
         // Mise à jour du dictionnaire et de Gamma
        [D,Gamma] = KSVD(D,X,Gamma);
        //disp(Gamma)
        //disp(D, "D=")
        //disp(j)
    end
endfunction


// -----------------------------------------------------------------------


// Codage parcimonieux

// Question 1

// 1. Apprentissage d'un nouveau dictionnaire de taille N*K (99*100)
// Gamma sera lui de taille K*l (100*108)
[D,Gamma] = Apprentissage_OMP(X,K,L);

// Ordre de parcimonie des vecteurs d'apprentissage
maxi=0;
for i=1:size(Gamma,"c")
    W=size(find(Gamma(:,i)<>0),"c");
    if maxi < W
        maxi=W; 
    end 
end
disp(maxi, "Ordre de parcimonie = ")
s=maxi; // s vaut donc 10

// 2. Méthode CoSaMP

//Algorithme du Compressive Sampling du Matching Pursuit - CoSaMP
function Alpha = CoSaMP(X, D, s)
    // D le dictionnaire
    // X un vecteur
    // s ordre de parcimonie
    
    K = size(D,2); // K représente le nombre de colonnes de D
    
    //Initialisation
    Epsilon = 10^(-6); // La précision souhaitée servant de valeur d'arrêt
    iter = 0; // Nombre d'itérations réalisées initialisé à 0
    residuel = X; //init du résiduel 0 égal au signal
    kmax = K/10; // Le nombre d'itérations maximum
    Alpha = zeros(K, 1); // Alpha est initialisée comme un vecteur de zéros 
    Supp = []; // Le support
    Supp1 = []; // Le support de sélection
    C = []; // Vecteur des contributions
    
    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((kmax > iter) & (norm(residuel) > Epsilon))
        // Etape de sélection, on calcule la contribution de chaque atome au résiduel dans C
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
        
        // Mise à jour du support tq supp=suppUsupp1
        if size(Supp)==[0,0]
            Supp=Supp1;
        else
            for i=1:size(Supp1,1)
                if Supp1(i)<>Supp
                    Supp($+1)=Supp1(i);
                end
            end
        end
        
        // Matrice des atomes actifs sélectionnés
        AS=D(:,Supp);
        
        // Estimation par la méthode des moindres carrés 
        Alpha(Supp)=pinv(AS' * AS) * AS' * X;
        
        // Rejet
        // On considère les s plus grands coefficients Z de Alpha
        if iter+1 ~= kmax
            Z=Alpha;
            newalpha=zeros(K,1);
            for i=1:s
                [val,mk] = max(Z);
                newalpha(mk) = val;
                Z(mk)=0;
            end
        
        // Alpha est notre nouveau vecteur composé de 0 et des s plus grands coefficients
           Alpha=newalpha;

            // On met à jour notre résiduel
            residuel = X - D * Alpha;
        end
        // On met à jour notre nombre d'itérations et on recommencer la boucle
        iter = iter + 1;
    end
endfunction


// Question 5 - Algorithme IRLS

// Algorithme IRLS (Iteratively reweighted least squares)
function alpha=IRLS(X,D,p)
    // X le vecteur du signal d'origine
    // D le dictionnaire
    // p p>0
    
    K = size(D,2); // K représente la taille du nbr de colonnes de D
    k = 0; // nbr itérations 
    kmax=K/10; // nbr itérations max
    Epsilon=0.1; // Coefficient de régularisation
    alpha=D'*pinv(D*D')*X; // Initialisation d'alpha 

    // Calcul des poids W
    W=(alpha.^2+Epsilon).^((p/2)-1);
    
    // Boucle générale pour calcule la prochaine itération Alpha^(k)
    for i=1:K
        // Calcul du nouvel alpha
        oldalpha=alpha;
        Q=diag(W); // Q est une matrice diagonale composée des poids 
        alpha=Q*D'*pinv(D*Q*D')*X;
        
        // Premier cas
        if ((abs(norm(alpha)-norm(oldalpha))>(sqrt(Epsilon)/100)) & (k<kmax))
            // On incrémente k et on retourne à l'étape 2 (calcul des poids)
            W=(alpha.^2+Epsilon).^((p/2)-1);
            k=k+1;
            break;
            
        // Second cas
        elseif ((abs(norm(alpha)-norm(oldalpha))<(sqrt(Epsilon)/100)) & (Epsilon<10^-8))
            // On modifie Epsilon
            Epsilon=Epsilon/10;
            
            // Si k ne dépasse pas le nbr d'itérations maximum
            if k<kmax
                // On incrémente k et on retourne à l'étape 2 (calcul des poids)
                W=(alpha.^2+Epsilon).^((p/2)-1);
                k=k+1;
                break;
            // Sinon on termine
            else 
                break;
            end
            
        // Dernier cas : on termine
        else
            break;
        end
    end
endfunction


// ---------------------------------------------------------------------


// Procédé du compressive sensing


// Matrices de mesures

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
    // P le pourcentage de mesures souhaitées
    // N taille du vecteur d'origine 
    
    // M le nombre de mesures considérées
    M=floor(P*N/100);
    p=0.5;
    phi=full(sprand(M,N,p));
endfunction


// Question 6

// 1. Calcul des vecteurs de mesure Yi
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


// Calcul pour chaque phi pour un P defini des vecteurs de mesure 
P1=phi1(30,N);
// Calcul des vecteurs de mesure pour phi1 pour un P defini
Y1=vecteurs_mesures(P1,X);
    
P2=phi2(30,N);
// Calcul des vecteurs de mesure pour phi2 pour un P defini
Y2=vecteurs_mesures(P2,X);
    
P3=phi3(30,N);
// Calcul des vecteurs de mesure pour phi3 pour un P defini
Y3=vecteurs_mesures(P3,X);
    
P4=phi4(30,N);
// Calcul des vecteurs de mesure pour phi4 pour un P defini
Y4=vecteurs_mesures(P4,X);

P5=phi5(30,N);
// Calcul des vecteurs de mesure pour phi5 pour un P defini
Y5=vecteurs_mesures(30,X);



// 2. Coherence mutuelle
// Pourcentage de mesure P variant entre 15 et 75
P=[15,20,25,30,50,75];

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


//// Initialisation des vecteurs contenant les coherences mutuelles en fonction de P   
//C_phi1=[]
//C_phi2=[]
//C_phi3=[]
//C_phi4=[]
//C_phi5=[]
//
//// Affichage de la cohérence mutuelle pour chaque P et chaque phi
//disp("Cohérence mutuelle")
//
//// Calcul de la cohérence mutuelle de chaque phi pour chaque P 
//for i = 1:length(P)
//    for j=1:1000
//        P1=phi1(P(i),N);
//        C1=coherence_mutuelle(P1,N,D);
//    end
//    C_phi1(i)=mean(C1);
//        
//    for j=1:1000
//        P2=phi2(P(i),N);
//        C2=coherence_mutuelle(P2,N,D);
//    end
//    C_phi2(i)=mean(C2);
//    
//    for j=1:1000
//        P3=phi3(P(i),N);
//        C3=coherence_mutuelle(P3,N,D);
//    end
//    C_phi3(i)=mean(C3);
//    
//    for j=1:1000
//        P4=phi4(P(i),N);
//        C4=coherence_mutuelle(P4,N,D);
//    end
//    C_phi4(i)=mean(C4);
//    
//    for j=1:1000
//        P5=phi5(P(i),N);
//        C5=coherence_mutuelle(P5,N,D);
//    end
//    C_phi5(i)=mean(C5);
//end
//
//
//disp(P, "P = ")
//disp(C_phi1', "Pour phi1, C=")
//disp(C_phi2', "Pour phi2, C=")
//disp(C_phi3', "Pour phi3, C=")
//disp(C_phi4', "Pour phi4, C=")
//disp(C_phi5', "Pour phi5, C=")
//
//// Illustration graphique de l'évolution de la cohérence mutuelle en fonction de P
//plot2d(P, [C_phi1 C_phi2 C_phi3 C_phi4 C_phi5], rect=[10,1,100,sqrt(N)])
//legends(["Phi1"; "Phi2" ; "Phi3" ; "Phi4" ; "Phi5"],[1,2,3,4,5],4)
//xtitle('Evolution de la cohérence mutuelle en fonction de P')


// Question 8 
// 1. A partir des 3 premiers vecteurs d'apprentissage, on reconstruit les vecteurs d'origine
// 2. Calcul de l'erreur relative

disp("Reconstruction vecteurs origines et calcul erreurs")

// On conserve uniquement phi2 et phi5 qui présentent la meilleure cohérence mutuelle

for i=1:3
    // Initialisation des vecteurs contenant les vecteurs reconstitués en fonction de P   
    E_phi2=zeros(5,size(P,1));
    E_phi4=zeros(5,size(P,1));
    
    // Calcul pour chaque méthode pour chaque P 
    for j = 1:length(P)
        
        // Methode MP
        meth=1;
            
        P2=phi2(P(j),N);
        // Calcul des vecteurs de mesure pour phi2 pour chaque P
        Y2=vecteurs_mesures(P2,X(:,i));
        // Reconstruction du signal
        A2=P2*D;
        alpha2=MP(Y2,A2);
        // Approximation du signal d'origine
        X2=D*alpha2;
        // Calcul d'erreur
        E2=norm(X(:,i)-X2)^2/norm(X(:,i))^2;
        // Stockage
        E_phi2(meth,j)=E2;
        
        
        P4=phi4(P(j),N);
        // Calcul des vecteurs de mesure pour phi4 pour chaque P
        Y4=vecteurs_mesures(P4,X(:,i));
        // Reconstruction du signal
        A4=P4*D;
        alpha4=MP(Y4,A4);
        // Approximation du signal d'origine
        X4=D*alpha4;
        // Calcul d'erreur
        E4=norm(X(:,i)-X4)^2/norm(X(:,i))^2;
        // Stockage
        E_phi4(meth,j)=E4;
        
        
        // Methode OMP
        meth=2;
        
        P2=phi2(P(j),N);
        // Calcul des vecteurs de mesure pour phi2 pour chaque P
        Y2=vecteurs_mesures(P2,X(:,i));
        // Reconstruction du signal
        A2=P2*D;
        alpha2=OMP(Y2,A2);
        // Approximation du signal d'origine
        X2=D*alpha2;
        // Calcul d'erreur
        E2=norm(X(:,i)-X2)^2/norm(X(:,i))^2;
        // Stockage
        E_phi2(meth,j)=E2;
        
        
        P4=phi4(P(j),N);
        // Calcul des vecteurs de mesure pour phi4 pour chaque P
        Y4=vecteurs_mesures(P4,X(:,i));
        // Reconstruction du signal
        A4=P4*D;
        alpha4=OMP(Y4,A4);
        // Approximation du signal d'origine
        X4=D*alpha4;
        // Calcul d'erreur
        E4=norm(X(:,i)-X4)^2/norm(X(:,i))^2;
        // Stockage
        E_phi4(meth,j)=E4;
        
        
        // Methode StOMP
        meth=3;
        
        P2=phi2(P(j),N);
        // Calcul des vecteurs de mesure pour phi2 pour chaque P
        Y2=vecteurs_mesures(P2,X(:,i));
        // Reconstruction du signal
        A2=P2*D;
        alpha2=StOMP(Y2,A2,2.2);
        // Approximation du signal d'origine
        X2=D*alpha2;
        // Calcul d'erreur
        E2=norm(X(:,i)-X2)^2/norm(X(:,i))^2;
        // Stockage
        E_phi2(meth,j)=E2;
        
        
        P4=phi4(P(j),N);
        // Calcul des vecteurs de mesure pour phi4 pour chaque P
        Y4=vecteurs_mesures(P4,X(:,i));
        // Reconstruction du signal
        A4=P4*D;
        alpha4=StOMP(Y4,A4,2.2);
        // Approximation du signal d'origine
        X4=D*alpha4;
        // Calcul d'erreur
        E4=norm(X(:,i)-X4)^2/norm(X(:,i))^2;
        // Stockage
        E_phi4(meth,j)=E4;
        

        // Methode CosaMP
        meth=4;
        
        
        P2=phi2(P(j),N);
        // Calcul des vecteurs de mesure pour phi2 pour chaque P
        Y2=vecteurs_mesures(P2,X(:,i));
        // Reconstruction du signal
        A2=P2*D;
        alpha2=CoSaMP(Y2,A2,s);
        // Approximation du signal d'origine
        X2=D*alpha2;
        // Calcul d'erreur
        E2=norm(X(:,i)-X2)^2/norm(X(:,i))^2;
        // Stockage
        E_phi2(meth,j)=E2;
        
        P4=phi4(P(j),N);
        // Calcul des vecteurs de mesure pour phi4 pour chaque P
        Y4=vecteurs_mesures(P4,X(:,i));
        // Reconstruction du signal
        A4=P4*D;
        alpha4=CoSaMP(Y4,A4,s);
        // Approximation du signal d'origine
        X4=D*alpha4;
        // Calcul d'erreur
        E4=norm(X(:,i)-X4)^2/norm(X(:,i))^2;
        // Stockage
        E_phi4(meth,j)=E4;
        
        
        // Methode IRLS
        meth=5;
        
        
        P2=phi2(P(j),N);
        // Calcul des vecteurs de mesure pour phi2 pour chaque P
        Y2=vecteurs_mesures(P2,X(:,i));
        // Reconstruction du signal
        A2=P2*D;
        alpha2=IRLS(Y2,A2,0.5);
        // Approximation du signal d'origine
        X2=D*alpha2;
        // Calcul d'erreur
        E2=norm(X(:,i)-X2)^2/norm(X(:,i))^2;
        // Stockage
        E_phi2(meth,j)=E2;
        
        P4=phi4(P(j),N);
        // Calcul des vecteurs de mesure pour phi4 pour chaque P
        Y4=vecteurs_mesures(P4,X(:,i));
        // Reconstruction du signal2
        A4=P4*D;
        alpha4=IRLS(Y4,A4,0.5);
        // Approximation du signal d'origine
        X4=D*alpha4;
        // Calcul d'erreur
        E4=norm(X(:,i)-X4)^2/norm(X(:,i))^2;
        // Stockage
        E_phi4(meth,j)=E4;
    end
end

disp("Exemple avec le vecteur apprentissage n°3")
disp(P, "P = ")
disp(E_phi2, "Erreur pour chaque P de chaque méthode pour phi2")
//disp(E_phi4)

// Illustration graphique de l'évolution de l'erreur en fonction de P pour chaque méthode
// phi2
//for i=1:5
//    plot2d(P, E_phi2(i,:), i, strf='021') 
//end
//legends(["MP"; "OMP" ; "StOMP" ; "CoSaMP" ; "IRLS"],[1,2,3,4,5],1);
//xtitle("Evolution de lerreur en fonction de P pour chaque méthode");
//
// phi4
//for i=1:5
//    plot2d(P, E_phi4(i,:), i, strf='021') 
//end
//legends(["MP"; "OMP" ; "StOMP" ; "CoSaMP" ; "IRLS"],[1,2,3,4,5],1);
//xtitle("Evolution de lerreur en fonction de P pour chaque méthode");



// ----------------------------------------------------------------


// Données manquantes 

disp("Données manquantes")

// Question 10
// 1. Reconstruction sans tenir compte des échantillones absents

function [Zr,Phi_r,V_choix]=Absence(X,r)
    // X le signal d'origine
    // r la nouvelle taille de l'echantillon
    
    // On choisit au hasard un vecteur de X
    V_r=X(:,ceil(size(X,2)*rand()));
    V_choix=V_r;
    
    // On choisit des éléments au hasard
    A=[] // Vecteur des positions des éléments manquants
    for j=1:r
       // On garde aléatoirement des valeurs dans V
       a=ceil(size(V_r,1)*rand());
       V_r(a)=[];
       A(j)=a;
    end
    
    // Calcul de la matrice Phi composée de 0 et de 1 lorsque l'élément est manquant
    Phi_r=zeros(r,size(X,1))
    for i=1:size(A,1)
        Phi_r(i,A(i))=1;
    end

    // Calcul de Zr
    Zr=Phi_r*V_choix
endfunction

// Vecteurs initialisés
Elements_manquants=[2,5,10,20,30,50,75];
Err=[];

// Reconstruction avec Zr et OMP

// Reconstruction du signal pour chaque cas
for i=1:length(Elements_manquants)
    r=N-Elements_manquants(i);
    // Calcul vecteur Zr et Phi
   [Zr,Phi,X_base]=Absence(X,r);
    A=Phi*D;
    alpha=OMP(Zr,A);
    // Approximation du signal d'origine
    Zn=D*alpha;
    // Calcul d'erreur et stockage
    Err(i)=norm(X_base-Zn)^2/norm(X_base)^2;
end

disp(Err', "Erreur de reconstruction pour Zr = ")

// 2. Reconstruction en tenant compte des échantillons absents

function [Z0,V_choix]=AbsenceZero(X,Zero)
    // X le signal d'origine
    // Zero le nbr d'echantillons à considérer nuls
    
    // On sélectionne un vecteur au hasard du signal d'origine
    V=X(:,ceil(size(X,2)*rand()));
    V_choix=V
    
    // On retire des éléments au hasard
    for j=1:Zero
       V(ceil(size(V,1)*rand()))=0;
    end
    
    Z0=V;
endfunction

// Initialisation des vecteurs
Zero=[2,5,10,20,30,50,75];
E2=[];
Phi4=phi4(100,N); 

// Reconstruction pour chaque cas
for j = 1:length(Zero)  
    // Calcul Z0 et Phi
    [Z0,X_base4]=AbsenceZero(X,Zero(j));
    // Reconstruction du signal
    Y4=vecteurs_mesures(Phi4,Z0)
    A4=Phi4*D;
    alpha4=OMP(Y4,A4);
    // Approximation du signal d'origine
    Zn4=D*alpha4;
    // Calcul d'erreur et stockage
    E2(j)=norm(X_base4-Zn4)^2/norm(X_base4)^2;
end
disp(E2',"Erreur de reconstruction pour Z0 = ")

