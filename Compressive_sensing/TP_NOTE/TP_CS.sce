// Execution du script des fonctions

// Path à modifier si emplacement du fichier different
load('dico.dat') // Renvoie un dictionnaire de taille 800 x 800
load('signalOriginal.dat') // Renvoie une matrice X de taille 800 x 1

// On prend tous l'espace nécessaire possible pour l'exécution du programme
stacksize('max')

// Verification parcimonieux
X = signalOriginal
D = dico 
alp = OMP(X,D)


// Etape 1
// a)

// Pourcentage de mesure P variant entre 10 et 100
P=[10,15,25,30,35,40,50,75,100]
   
// Initialisation des vecteurs contenant les coherences mutuelles en fonction de P    
C_phi1=[]
C_phi2=[]
C_phi3=[]
C_phi4=[]
C_phi5=[]

// Calcul de la cohérence mutuelle de chaque phi pour chaque P 
for i = 1:length(P)
    P1=phi1(P(i),N)
    C1=coherence_mutuelle(P1,N,D)
    C_phi1(i)=C1
    
    P2=phi2(P(i),N)
    C2=coherence_mutuelle(P2,N,D)
    C_phi2(i)=C2
    
    P3=phi3(P(i),N)
    C3=coherence_mutuelle(P3,N,D)
    C_phi3(i)=C3
    
    P4=phi4(P(i),N)
    C4=coherence_mutuelle(P4,N,D)
    C_phi4(i)=C4
    
    P5=phi5(P(i),N)
    C5=coherence_mutuelle(P5,N,D)
    C_phi5(i)=C5
end

// Affichage de la cohérence mutuelle pour chaque P et chaque phi
disp("Cohérence mutuelle")

disp(P, "P = ")

disp(C_phi1', "Pour phi1, C=")
disp(C_phi2', "Pour phi2, C=")
disp(C_phi3', "Pour phi3, C=")
disp(C_phi4', "Pour phi4, C=")
disp(C_phi5', "Pour phi5, C=")

// Illustration graphique de l'évolution de la cohérence mutuelle en fonction de P
plot2d(P, [C_phi1 C_phi2 C_phi3 C_phi4 C_phi5], rect=[10,1,100,sqrt(N)])
legends(["Phi1"; "Phi2" ; "Phi3" ; "Phi4" ; "Phi5"],[1,2,3,4,5],4)
xtitle('Evolution de la cohérence mutuelle en fonction de P')


// b)

P=25
Y1=vecteurs_mesures(phi1(P,N),X)
Y2=vecteurs_mesures(phi2(P,N),X)
Y3=vecteurs_mesures(phi3(P,N),X)
Y4=vecteurs_mesures(phi4(P,N),X)
Y5=vecteurs_mesures(phi5(P,N),X)


// Etape 2 - Reconstruction du signal
// a)
A1=phi1(P,N)*D
A2=phi2(P,N)*D
A3=phi3(P,N)*D
A4=phi4(P,N)*D
A5=phi5(P,N)*D

alpha1=OMP(Y1,A1)
alpha2=OMP(Y2,A2)
alpha3=OMP(Y3,A3)
alpha4=OMP(Y4,A4)
alpha5=OMP(Y5,A5)

// b) Approximation

X1=D*alpha1
X2=D*alpha2
X3=D*alpha3
X4=D*alpha4
X5=D*alpha5





//subplot(2,3,1)
id=winsid()
old=id($)
new=old+1
f=scf(new)
plot2d(X,rect=[0,-2,800,2])
id=winsid()
old=id($)
new=old+1
f=scf(new)
//subplot(2,3,2)
plot(X1)
id=winsid()
old=id($)
new=old+1
f=scf(new)
//subplot(2,3,3)
plot(X2)
id=winsid()
old=id($)
new=old+1
f=scf(new)
subplot(2,3,4)
plot(X3)
id=winsid()
old=id($)
new=old+1
f=scf(new)
//subplot(2,3,5)
plot(X4)
id=winsid()
old=id($)
new=old+1
f=scf(new)
//subplot(2,3,6)
plot(X5)


M1=MSE(X1,X)
M2=MSE(X2,X)
M3=MSE(X3,X)
M4=MSE(X4,X)
M5=MSE(X5,X)
