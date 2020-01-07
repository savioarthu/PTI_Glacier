// Taille du vecteur d'origine
N=99

// Matrice aléatoire générée à partir d'un processus uniformément distribuée
function phi=phi1(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100)
    phi=rand(M,N)
endfunction

// Matrice aléatoire générée à partir d'un processus bernoullien {-1,1}
function phi=phi2(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100)
    p=0.5
    phi=grand(M,N,"bin",1,p)*2-1
endfunction

// Matrice aléatoire générée à partir d'un procesus bernoullien {0,1}
function phi=phi3(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100)
    p=0.5
    phi=grand(M,N,"bin",1,p)
endfunction

// Matrice aléatoire générée à partir d'un processus gaussien identique et idépendamment distribué (i.i.d) avec une moyenne nulle et une variance 1/M : N(0,1//M)
function phi=phi4(P,N)
     // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
    
    // M le nombre de mesures considérées
    M=floor(P*N/100)
    phi=grand(M,N,"nor",0,sqrt(1/M))
endfunction

// Matrice creuse ou parcimonieuse générée de façon indépendante 
function phi=phi5(P,N)
    // P : Le pourcentage de mesures souhaitées
    // N : taille du vecteur d'origine
     
    // M le nombre de mesures considérées
    M=floor(P*N/100)
    p=0.5
    phi=full(sprand(M,N,p))
endfunction

// Algorithme de la cohérence mutuelle
function C=coherence_mutuelle(phi,N,D)
    // phi : matrice de mesure
    // N : taille du vecteur d'origine
    // D : dictionnaire d'apprentissage
    
    // Initialisation du maximum
    maxi=0
    
    // La cohérence mutuelle est égale à la plus grande valeur absolue du produit scalaire entre les vecteurs colonnes de D et les vecteurs lignes de phi
    for i=1:size(phi,"r")
        for j=1:size(D,"c")
            if i<>j
                if maxi< abs(phi(i,:)*D(:,j))/(norm(phi(i,:))*norm(D(:,j)))
                    maxi=max(abs(phi(i,:)*D(:,j))/(norm(phi(i,:))*norm(D(:,j))))
                end
            end
        end
    end
    
    C=sqrt(N)*maxi
    // La cohérénce mutuelle doit être entre  1 et sqrt(N)
endfunction


function C=coherence(A)
    // A : A=Phi*D une matrice
    
    // Initialisation du maximum
    maxi = 0
    
    // La cohérence de A est égale à la plus grande valeur absolue du produit scalaire entre deux vecteurs colonnes distincts de A
    for i = 1:size(A,"r")
        for j =1:size(A,"r")
            if i<>j
                if maxi< abs(A(:,i)'*A(:,j))/(norm(A(:,i))*norm(A(:,j)))
                    maxi=max(abs(A(:,i)'*A(:,j))/(norm(A(:,i))*norm(A(:,j))))
                end
            end
        end
    end
    C=maxi
endfunction


function Z=vecteurs_mesures(phi,X)
    l=size(X,"c")
    Z=[]
    for i = 1:l
        Z(:,i)=phi*X(:,i)
    end
endfunction

function M=MSE(A,B)
    n=size(A,"r")
    for i=1:n
        M=(A(i)-B(i))^2
    end
    M=(1/n)*M
endfunction


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
//    P1=phi1(P(i),N)
//    C1=coherence_mutuelle(P1,N,D)
//    C_phi1(i)=C1
//    
//    P2=phi2(P(i),N)
//    C2=coherence_mutuelle(P2,N,D)
//    C_phi2(i)=C2
//    
//    P3=phi3(P(i),N)
//    C3=coherence_mutuelle(P3,N,D)
//    C_phi3(i)=C3
//    
//    P4=phi4(P(i),N)
//    C4=coherence_mutuelle(P4,N,D)
//    C_phi4(i)=C4
//    
//    P5=phi5(P(i),N)
//    C5=coherence_mutuelle(P5,N,D)
//    C_phi5(i)=C5
//end
//
//// Affichage de la cohérence mutuelle pour chaque P et chaque phi
//disp("Cohérence mutuelle")
//
//disp(P, "P = ")
//
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
//
//
//
//// Calcul de la cohérence de A=phi*D
//
//// Initialisation des vecteurs contenant les coherences de A en fonction de P    
//C_phi1=[]
//C_phi2=[]
//C_phi3=[]
//C_phi4=[]
//C_phi5=[]
//
//// Calcul de la cohérence de A pour chaque phi en fonction de P
//for i = 1:length(P)
//    mu1=coherence(phi1(P(i),N)*D)
//    C_phi1(i)=mu1
//    
//    mu2=coherence(phi2(P(i),N)*D)
//    C_phi2(i)=mu2
//    
//    mu3=coherence(phi3(P(i),N)*D)
//    C_phi3(i)=mu3
//    
//    mu4=coherence(phi4(P(i),N)*D)
//    C_phi4(i)=mu4
//    
//    mu5=coherence(phi5(P(i),N)*D)
//    C_phi5(i)=mu5
//end
//
//// Affichage de la cohérence de A pour chaque P et chaque phi
//disp("Cohérence de A")
//
//disp(P, "P = ")
//
//disp(C_phi1', "Pour phi1, C=")
//disp(C_phi2', "Pour phi2, C=")
//disp(C_phi3', "Pour phi3, C=")
//disp(C_phi4', "Pour phi4, C=")
//disp(C_phi5', "Pour phi5, C=")
//
//P=25
//Y1=vecteurs_mesures(phi1(P,N),XX)
//Y2=vecteurs_mesures(phi1(P,N),XX)
//Y3=vecteurs_mesures(phi1(P,N),XX)
//Y4=vecteurs_mesures(phi1(P,N),XX)
//Y5=vecteurs_mesures(phi1(P,N),XX)
//
