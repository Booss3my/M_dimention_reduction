from sklearn.neighbors import KNeighborsClassifier
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

from PCA import PCA

#loader la base 
mnist = MNIST('../samples')
x_train, y_train=mnist.load_training()

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)  #pas utile en PCA

#constantes 
training_num=4000
test_num=2000
err=[]
for coeff in range(1,10):
    print(coeff) 

    #reduction de dimension
    #pca
    d_pca=PCA(x_train[:test_num+training_num+1,:],coeff)  #données réduites 

    #lda


    #classification
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(d_pca[:training_num,:], y_train[:training_num]) #utiliser les 4000 premiers points comme repères pour classer le reste
    y_pred=classifier.predict(d_pca[training_num+1:test_num+training_num+1,:])

    #evaluer l'erreur 
    nb_erreur=0
    for i in range(0,len(y_pred)):
        if y_pred[i]!=y_train[training_num+i+1]:
            nb_erreur+=1

    err.append((nb_erreur/test_num)*100)

#affichage
plt.plot([i for i in range(1,10)],err)
plt.xlabel("Coefficient de réduction")
plt.ylabel("erreur de classification en %")
plt.show()

