from sklearn.neighbors import KNeighborsClassifier
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from time import time
from PCA import PCA
from LDA import LDA
#loader la base 
mnist = MNIST('../samples')
x_train, y_train=mnist.load_training()

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)  #pas utile en PCA

#constantes 
training_num=4000
test_num=2000
errp=[]
errl=[]
p_time=[]
for coeff in range(1,10):

    #reduction de dimension
    #pca
    d_pca=PCA(x_train[:test_num+training_num+1,:],coeff)  #données réduites 

   
    #classification
    ptime=time()
    classifier1 = KNeighborsClassifier(n_neighbors=5)
    classifier1.fit(d_pca[:training_num,:], y_train[:training_num]) #utiliser les 4000 premiers points comme repères pour classer le reste
    y_predP=classifier1.predict(d_pca[training_num+1:test_num+training_num+1,:])

    p_time.append(time()-ptime)

    #evaluer l'erreur 
    nb_erreurp=0
    nb_erreurl=0
    for i in range(0,len(y_predP)):
        if y_predP[i]!=y_train[training_num+i+1]:
            nb_erreurp+=1
    err_p=(nb_erreurp/test_num)*100
    errp.append(err_p)

    print('erreur PCA',err_p)
    print('reduction en %', 100/coeff)
    print('temps de classfication ',ptime)

 #lda
d_lda=LDA(x_train[:test_num+training_num+1,:],y_train[:test_num+training_num+1])
ltime=time()
classifier2 = KNeighborsClassifier(n_neighbors=5)
classifier2.fit(d_lda[:training_num,:], y_train[:training_num])
y_predL=classifier2.predict(d_lda[training_num+1:test_num+training_num+1,:])
ltime=time()-ltime

 #evaluer l'erreur 

nb_erreurl=0
for i in range(0,len(y_predL)):
    if y_predL[i]!=y_train[training_num+i+1]:
        nb_erreurl+=1

err_l=(nb_erreurl/test_num)*100
print('erreur LDA',err_l)
print('reduction de dimention en %', 900/x_train.shape[1])
print('temps de classfication ',ltime)
#affichage
plt.plot([1/i for i in range(1,10)],errp,label='PCA: 784/coeff')
plt.plot([1/i for i in range(1,10)],[err_l for i in range(1,10)],label='LDA: 9')

plt.xlabel("(PCA)taux de réduction  ")
plt.ylabel("taux d'erreur de classification en %")
plt.show()

plt.plot([1/i for i in range(1,10)],p_time,label='PCA: 784/coeff')
plt.plot([1/i for i in range(1,10)],[ltime for i in range(1,10)],label='LDA: 9')

plt.xlabel("(PCA)taux de réduction")
plt.ylabel("temps de classification")
plt.show()

