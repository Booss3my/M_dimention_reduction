from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import math as mt

mnist = MNIST('../samples')
x_train, y_train=mnist.load_training()

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)  #pas utile en PCA

print('Dimensions: %s x %s' % (x_train.shape[0], x_train.shape[1]))
print(x_train[0])

X=np.transpose(x_train)

Mat_corr = np.dot(X,np.transpose(X))

print('Dimensions: %s x %s' % (Mat_corr.shape[0], Mat_corr.shape[1])) 

nouv_dim=mt.floor(Mat_corr.shape[0]/3)  #dimention reduite

