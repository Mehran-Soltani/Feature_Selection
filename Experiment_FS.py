


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from keras.datasets import mnist
from scipy.io import loadmat 
#from utils import getSyntheticDataset
import os
from scipy.special import gamma,psi
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical   
from keras.callbacks import EarlyStopping
from scipy import ndimage
from scipy.linalg import det
from numpy import pi
import pickle
import scipy.io
from sklearn import svm
#from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def test_kmeans(x_test, y_test, faetures, number=20):
    x = x_test.reshape(len(x_test), -1)
    x = x[:,faetures]
    n_clusters = len(set(y_test))
    res = np.zeros(number)
    for index in range(number):
        km = KMeans(n_clusters=n_clusters, n_jobs=4, init='random')
        y = km.fit_predict(x)
        cost = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                cost[i,j] = -sum(np.logical_and(y==i,y_test==j))
        row_ind, col_ind = linear_sum_assignment(cost)
        y_changed = 100*np.ones_like(y)
        for i,j in zip(row_ind, col_ind):
            y_changed[y==i] = j
        res[index] = float(np.sum(y_changed==y_test))*100./float(len(y))
    return (np.mean(res), np.std(res))

def test_knn(x_train, y_train, x_test, y_test, features):
    x_test = x_test.reshape(len(x_test), -1)
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test[:,features]
    x_train = x_train[:,features]
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
    knn.fit(x_train, y_train)
    y = knn.predict(x_test)
    return np.sum(y==y_test)*100/len(y)

def test_softmax(x_train, y_train, x_test, y_test, features):
    early_stopping = EarlyStopping(patience=2)
    x_test = x_test.reshape(len(x_test), -1)
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test[:,features]
    x_train = x_train[:,features]
    num_classes = len(np.unique(y_train))
    model = Sequential([Dense(num_classes, input_dim = len(features), activation='softmax')])
    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    y_binary_train = to_categorical(y_train, num_classes=num_classes)
    y_binary_test = to_categorical(y_test, num_classes=num_classes)
    model.fit(x_train, y_binary_train, epochs=1000, validation_split=.15, verbose=0)
    result = model.evaluate(x_test, y_binary_test)
    return result[1]


def nearest_distances(X, k=1):
        '''
        X = array(N,M)
        N = number of points
        M = number of dimensions
        returns the distance to the kth nearest neighbor for every point in X
        '''
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
        return d[:, -1] # returns the distance to the kth nearest neighbor
    

def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))


def entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))

def maxrel_minred(x_train , weights , N , Bestfea_nums):
    
    # N select the N wieghts with max values 
    ind1 = np.argpartition(weights, -N)[-N:]
    selectedN = x_train[:,ind1]
        
    MI_scores = np.zeros(N)
    
    for i in range(N):  
        X1 = selectedN[:,i]  
        X2 = np.delete(selectedN, i , 1)
        X1.reshape(-1,1)
        X_z = np.zeros(X1.shape[0])
        h = np.column_stack((X1,X_z))
        d1 = nearest_distances(h , k = 15)
        d2 = nearest_distances(X2 , k = 15)
        MI_scores[i] = (np.sum(np.log(d1))+(99)*np.sum(np.log(d2)))/180
    
    
    ind2 = np.argpartition(MI_scores, -1)[-1:]
    

    bestfea_ind = ind2
    
    alpha = 0.5
    phi = np.array([]) # max information - min redundancy score
    mi = np.array([])
    
    for i in range(Bestfea_nums-1):
        phi = np.array([])
        for j in range(N):
            if j in bestfea_ind:
                phi = np.append(phi , - float('inf'))
                
            else:
                 X1 = selectedN[:,j]
                 #dummy_ind = np.append(bestfea_ind, j)
                 X2 = selectedN[:,bestfea_ind]
                 X_z = np.zeros(X1.shape[0])
                 
                 # concat a zero vector with data sets
                 h1 = np.column_stack((X1,X_z))
                 h2 = np.column_stack((X2,X_z))
                 d1 = nearest_distances(h1 , k = 15)
                 d2 = nearest_distances(h2 , k = 15)
                 miscor = (np.sum(np.log(d1))+(len(bestfea_ind)-1)*np.sum(np.log(d2)))/180
                 mi = np.append(mi,miscor)
                 
                 
                 phi = np.append(phi , (MI_scores[j] + alpha*mi[-1]))
                 
            
        
        new_ind = np.argpartition(phi, -1)[-1:]
        bestfea_ind = np.append(bestfea_ind,new_ind)
    
    
    finalbest_ind_mi = ind1[(bestfea_ind)]
    finalbest_ind_aefs = np.argpartition(weights, -Bestfea_nums)[-Bestfea_nums:]
    return finalbest_ind_mi , finalbest_ind_aefs 

if __name__ == '__main__':

    
    alpha = 0.001
    beta = 0.01 

    dataset = 'face'

    if(dataset == 'face'):
        data = loadmat('E:/MSc/TensorFlow Learn/AEFS/warpPIE10P.mat')
        X = data['X']/255.
        #x_train = X[0:180,]
        #x_test = X[180:209,]
        Y = data['Y']-1
        #y_train = Y[0:180]
        #y_test = Y[180:209]
        n_splits = 10
        kf = KFold(n_splits)
        input_shape = (44,55)
        input_dim = 44*55
        hidden = 128
        
        # load weights 
        weights = np.load('E:/MSc/TensorFlow Learn/AEFS/weights.npy')
        N = 100
        Bestfea_nums = 40
        acc_mi = np.zeros(n_splits)
        acc_aefs = np.zeros(n_splits)
        counter = 0
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
                
            [fea_idx_mi , fea_idx_aefs] = maxrel_minred(x_train , weights , N , Bestfea_nums)
        
            mi_mask =np.zeros(input_dim) 
            mi_mask[fea_idx_mi] = 1
            bestfea_mi_image = np.multiply(mi_mask, weights)
            
            aefs_mask = np.zeros(input_dim) 
            aefs_mask[fea_idx_aefs]=1
            
            bestfea_aefs_image = np.multiply(aefs_mask , weights)
            
            fig = plt.figure(figsize=(10,3))
            ax = fig.add_subplot(131)
            weights_martrix = weights.reshape(44,55)
            ax.imshow(weights_martrix)
            ax = fig.add_subplot(132)
            ax.imshow(bestfea_mi_image.reshape(44,55))  
            ax = fig.add_subplot(133)
            ax.imshow(bestfea_aefs_image.reshape(44,55))
            
            
            ## evalute the feature selection results of AEFS + MI 
            clf_mi = svm.LinearSVC()    # linear SVM
            clf_aefs = svm.LinearSVC()
            
            
            clf_mi.fit(x_train[:, fea_idx_mi ], y_train)
            clf_aefs.fit(x_train[:, fea_idx_aefs ], y_train)
            
            
            y_predict_mi = clf_mi.predict(x_test[:,fea_idx_mi])
            y_predict_aefs = clf_aefs.predict(x_test[:,fea_idx_aefs])
        
            # obtain the classification accuracy on the test data
            
            acc_mi[counter] = accuracy_score(y_test, y_predict_mi)
            acc_aefs[counter] = accuracy_score(y_test , y_predict_aefs)
            counter = counter+1
            #print('mi acc' , acc_mi)
            #print('aefs_acc' , acc_aefs)
        
    
    