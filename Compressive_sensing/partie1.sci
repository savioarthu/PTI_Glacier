// Mini projet script.sci
// Groupe Thibault LEDOUX - Arthur SAVIO - Pierre Soulier

clear
clc

// Données necessaires 
// Path à modifier si emplacement du fichier different
load('data.dat')
// Renvoie une matrice XX de taille N*l avec N=99 et l =108

// Redefinition de fonctions
// funcprot(0)

// On prend tous l'espace nécessaire possible pour l'exécution du programme
stacksize('max')

// Taille de X
X=XX; // Matrice composée des vecteurs d'apprentissage Xi
N=99; // Lignes de X
l=108; // Colonnes de X
K=100; // Taille souhaitée du dictionnaire
L=10; // Nombre d'itérations



// Fonctions nécessaires préalables au mini projet


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
    C = zeros(K,1); // Vecteur des valeurs max de chaque colonne de D
    P = [] // P représente l'ensemble des indices des coefficients. Il est vide à l'étape 0.

    // A l'étape k>=1, on boucle tant que notre nombre d'itérations n'ont pas dépassé un seuil prédéfini et que notre critère d'arrêt est supérieur à un seuil epsilon prédéfini
    while ((kmax > iter) & (norm(residuel) > Epsilon))
        // Sélection de l'atome identique au MP : celui qui contribue le plus au résiduel R^(0
        for i = 1:K
            if (norm(D(:,i))<>0)
                C(i) = norm(D(:,i)' * residuel) / norm(D(:,i));
            end
        end
        // Ainsi on retient toujours l'indice correspondant au maximum 
        [val,mk] = max(abs(C));
        //disp(mk) // Pour vérifier l'atome sélectionné
        
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
         // Mise à jour du dictionnaire de de Gamma
        [D,Gamma] = KSVD(D,X,Gamma);
        //disp(Gamma)
        //disp(D, "D=")
        //disp(j)
    end
endfunction

// Codage parcimonieux

// Question 1

// 1. Apprentissage d'un nouveau dictionnaire de taille N*K (99*100)
// Gamma sera lui de taille K*l (100*108)
[D,Gamma] = Apprentissage_OMP(X,K,L);

// Ordre de parcimonie des vecteurs d'apprentissage
max=0;
for i=1:size(Gamma,"c")
    W=size(find(Gamma(:,i)<>0),"c");
    if max < W
        max=W; 
    end 
end
disp(max)
s=max; // s vaut donc 10
