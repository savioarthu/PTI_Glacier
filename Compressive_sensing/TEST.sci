function Zr=Absence(X,Abs)
    // X le signal d'origine
    // Abs le nbr d'echantillons absents
    
    // On sélectionne un vecteur au hasard du signal d'origine
    V=X(:,ceil(size(X,2)*rand()))
    
    // On retire des éléments au hasard
    for j=1:Abs
       V(ceil(size(V,1)*rand()))=[]
    end
    Zr=V
endfunction



Abs=10;
Zr=Absence(X,Abs);
disp(size(Zr))
