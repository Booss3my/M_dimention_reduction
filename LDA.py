from sklearn.neighbors import KNeighborsClassifier
from mnist import MNIST
import matplotlib.pyplot as plt
import math as mt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def LDA(x_train,y_train):

    # # lda
    clf = LinearDiscriminantAnalysis()
    x_new=clf.fit_transform(x_train,y_train)
    
    return(x_new)




# dimention reduite

# nouv_dim=mt.floor(d.shape[0]/coeff) 

# calcul des moyennes

# moyennes=np.zeros((d.shape[0],10))
# num=[0 for i in range(0,10)]

# print('moyenne Dimensions: %s x %s' % (moyennes.shape[0], moyennes.shape[1]))

# for i in range(d.shape[1]):
#     moyennes[:,y_train[i]]+=d[:,i]  
#     num[y_train[i]]+=1

# moyenne_tot=moyennes[:,0]
# for i in range(1,10):
#     moyenne_tot+=moyennes[:,i]
#     moyennes[:,i]=moyennes[:,i]/num[i]

# moyenne_tot=moyenne_tot/d.shape[1]

# print(num)
# matrices de covariance
# S=np.zeros((d.shape[0],d.shape[0],10)) #contient les matrices de covariance S[:,:,i] 

# for i in range(d.shape[1]):
#     scat=d[:,i]-moyennes[:,y_train[i]]
#     S[:,:,y_train[i]]+=np.dot(scat,np.transpose(scat))

# within_class scatter 
# S_w=np.zeros((d.shape[0],d.shape[0]))
# S_b=np.zeros((d.shape[0],d.shape[0]))
# for i in range(10):
#     S_w+=S[:,:,i]*((num[i]-1)/d.shape[1]-1)
    
#     diff_mo=moyennes[:,i]-moyenne_tot
#     S_b+=num[i]*np.dot(diff_mo,np.transpose(diff_mo))    

# print(np.linalg.det(S_w))
# Mat=np.dot(np.linalg.inv(S_w),S_b)

# w,v = np.linalg.eig(Mat)  #w tableau des valeurs propres repétés le nombre de leur multiplicités , v[:,i] vecteur propre associé à w[i] 

# v=np.real(v)

# print('Dimensions: %s x %s' % (v.shape[0], v.shape[1])) 

# changement de base (v la matrice de passage)
# d_nv=np.dot(v,d)

# données réduites 
# d_reduites=d_nv[1:nouv_dim,:]

# print('reduced Dimensions: %s x %s' % (d_reduites.shape[0], d_reduites.shape[1]))



