from mlxtend.data import loadlocal_mnist
import platform

if not platform.system() == 'Windows':
    X, y = loadlocal_mnist(
            images_path='train-images-idx3-ubyte', 
            labels_path='train-labels-idx1-ubyte')

else:
    X, y = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')

mean_vectors = []
