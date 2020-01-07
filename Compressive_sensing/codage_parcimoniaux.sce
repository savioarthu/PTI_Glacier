// Codage Matching Pursuit

// Execution du script des fonctions

// A modifier si emplacement different
exec('H:/Documents/COURS/ING2/SEMESTRE2/Compressive_sensing/codage_parcimoniaux.sci')
load('H:/Documents/COURS/ING2/SEMESTRE2/Compressive_sensing/data.dat')

stacksize('max')

// Paramètres nécessaires à l'exécution
Epsilon=10^-6 
kmax=100 // Nbr itérations maximum

// Premier test avec un dictionnaire D1 et un vecteur X1
D1=[1/2*sqrt(2) 1/3*sqrt(3) 1/3*sqrt(6) 2/3 -1/3; -1/2*sqrt(2) -1/3*sqrt(3) -1/6*sqrt(6) 2/3 -2/3; 0 -1/3*sqrt(3) 1/6*sqrt(6) 1/3 2/3]
X1=[4/3-sqrt(2)/2 ; 4/3+sqrt(2)/2 ; 2/3]
 
// On détermine une représentation parcimonieuse de X1 dans le dictionnaire D1 avec le principe du Matching Puirsuit dans un premier temps puis avec le principe de l'OMP dans un second temps.
A=MP(D1,X1,kmax,Epsilon)
Alpha=OMP(D1,X1,kmax,Epsilon)
Alph=StOMP(D1,X1,kmax,Epsilon)

// Les résultats sont exactement les mêmes avec les deux méthodes pour ce cas.


// Second test avec un dictionnaire D2 et un vecteur X2
D2=[1 1 2 5 0 0 3 -2 1 2 2 2; 0 -1 -1 1 0 0 5 0 2 2 7 -1; 1 1 1 5 1 2 2 1 1 1 1 5; 1 5 2 2 5 0 -4 5 1 5 0 0; 0 2 2 1 1 0 0 0 0 4 -1 -2; -1 2 2 2 -2 -3 -4 1 1 1 1 0]
X2=[-10;-10;1;21;0;9]

// On détermine une représentation parcimonieuse de X2 dans le dictionnaire D2 avec le principe du Matching Puirsuit dans un premier temps puis avec le principe de l'OMP dans un second temps.
A2=MP(D2,X2,kmax,Epsilon)
Alpha2=OMP(D2,X2,kmax,Epsilon)
Alph2=StOMP(D2,X2,kmax,Epsilon)

// Les résultats sont bien différents entre les deux méthodes. On remarque que l'OMP est meilleure car on obtient plus de zéro qu'avec le MP.



//Algorithme d'apprentissage d'un dictionnaire par KSVD
k=100
kmax=10
Epsilon=10^-6
L=10
X=grand(3,10,"uin",1,4)

[D,Gamma]=Apprentissage(XX,k,L,Epsilon,kmax)
a=PSNR(D*Gamma,100)
b=PSNR(X,100)
c=abs(b-a)

L=1
[D2,Gamma2]=Apprentissage(XX,k,L,Epsilon,kmax,StOMP)

