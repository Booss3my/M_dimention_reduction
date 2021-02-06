
import matplotlib.pyplot as plt
import math as mt
import numpy as np

def PCA(x_train,coeff):
    

    d=np.transpose(x_train) #donnees


    ## pca

    print('original Dimensions: %s x %s' % (d.shape[0], d.shape[1]))

    Mat_corr = np.dot(d,np.transpose(d))

    #dimention reduite

   
    nouv_dim=mt.floor(Mat_corr.shape[0]/coeff) 

    w,v = np.linalg.eig(Mat_corr)  #w tableau des valeurs propres repétés le nombre de leur multiplicités , v[:,i] vecteur propre associé à w[i] 

    v=np.real(v)

    #print('Dimensions: %s x %s' % (v.shape[0], v.shape[1])) 

    #changement de base (v la matrice de passage)
    d_nv=np.dot(v,d)

    #données réduites 
    d_reduites=d_nv[1:nouv_dim,:]
    
    print('reduced Dimensions: %s x %s' % (d_reduites.shape[0], d_reduites.shape[1]))

    #print(np.dot(np.transpose(v[:,:nouv_dim]),v[:,:nouv_dim])) #(verification)matrice identité (On remarque orthogonalité)
    return(np.transpose(d_reduites)) 