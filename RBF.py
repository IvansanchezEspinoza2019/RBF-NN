# Libraries
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

problem = 3   # Choose the problem

if problem == 1: # Regression
    ####### Create regression problem ########
    p = 50   # sample size
    x = np.linspace(-5, 5, p).reshape(1, -1)
    y = 2 * np.cos(x) + np.sin(3*x) +5

    # Plot points
    plt.plot(x, y, 'or')

    # Number of neurons
    k = 30

    ########## Unsupervised training ##################
    model = KMeans(n_clusters=k)
    model.fit(x.T)

    # Get the centroid of every neuron
    c =  model.cluster_centers_

    # Calculate Sigma
    sigma = ((max(c) - min(c))/np.sqrt(2*k))
    sigma = sigma[0]

    # Calculate G matrix
    G = np.zeros((p, k))
    for i in range(p):
        for j in range(k):
            # euclidean distance
            dist = np.linalg.norm(x[0,i] - c[j], 2)
            # calculate Gaussian function
            G[i, j] = np.exp((-1/sigma**2)*dist**2)

    ######  Supervised trainning ###########
    W = np.dot(np.linalg.pinv(G), y.T) # calculate pseudo-inverse of W 


    ##### Line regression #######

    ### make the line regression ##
    p = 200  # points of the line
    xnew =  np.linspace(-5, 5, p).reshape(1, -1)

    # Propagation in the first layer 
    G = np.zeros((p, k))
    for i in range(p):
        for j in range(k):
            dist = np.linalg.norm(xnew[0,i] - c[j], 2)
            G[i, j] = np.exp((-1/sigma**2) * dist**2) 

    ### Prediction ##
    ynew = np.dot(G, W)

    # plot the regresions over the points
    plt.title("RBF")
    plt.plot(xnew.T, ynew, '-b', label="Regression")
    plt.legend()
    plt.grid()
    plt.show()
    
elif problem==2: # Regresion
    
    #### Load data ###
    data = pd.read_csv("regresion_rbfnn.csv")
    x = np.asanyarray(data.iloc[:, 0]).reshape(1, -1)
    y = np.asanyarray(data.iloc[:, 1]).reshape(1, -1)
    
    ## Number of RBF neurons ##
    k = 50
    
    ## plot dataset ##
    plt.plot(x, y, 'ro', markersize=2)
    
    ## Clustering algorithm ###
    model = KMeans(n_clusters=k)
    
    ### Unsupervised training ###
    model.fit(x.T)
    
    ### Centroid of each RBF neuron ###
    c = model.cluster_centers_
   
    ### Gaussian function ###
    
    # calculate sigma based on neurons centroids
    sigma =  ((max(c) - min(c)) / np.sqrt(2*k))[0]
    
    # Create G matrix (The outputs of every neuron for each pattern)  shape: (N. samples of X, k neurons)
    G = np.zeros((x.shape[1], k))
    
    for i in range(x.shape[1]):
        for j in range(k):
            """
                Gaussian function
            """
            distance = np.linalg.norm(x[0, i] - c[j], 2)   # norm two (euclidian norm)
            G[i, j] = np.exp((-1/sigma**2) * distance**2)  # save it on the matrix
            
    ### Supervised training ###
    # pseudo-inverse
    W = np.dot(np.linalg.pinv(G), y.T)
    
    #### prediction ####
    # points for doing the linear regression
    xnew = np.linspace(0, 10, 500).reshape(1, -1)
    
    # same: calculate the Gaussiaon formula 
    G  = np.zeros((xnew.shape[1], k))
    
    for i in range(xnew.shape[1]):
        for j in range(k):
            distance = np.linalg.norm(xnew[0, i] - c[j], 2)
            G[i,j] = np.exp((-1/sigma**2) * distance**2)
    
    # make predictions
    ynew = np.dot(G, W)
    
    # plot the regresions over the points
    plt.title("RBF")
    plt.plot(xnew.T, ynew, '-b', label="Regression")
    plt.legend()
    plt.grid()
    plt.show()

elif problem == 3:   ### Clasification
    
    def plotData(x, y):
        ## plot every single pattern of every class in the graph 
        for i in range(x.shape[1]):
            if y[0, i] == 1:
                plt.plot(x[0, i], x[1, i], '.b', markersize=2)   # class 0
            else:
                plt.plot(x[0, i], x[1, i], '.r', markersize=2)   # class 1
        
    
    ######## Read dataset ########
    data = pd.read_csv("moons.csv")
    
    # get x and y
    x = np.asanyarray(data.drop(columns=['y']))
    y = np.asanyarray(data[['y']])
    
    ## number of clusters (RBF neurons) ##
    k = 17
    
    ####### Unsupervised training #######
    model = KMeans(n_clusters=k)
    
    # train kmeans
    model.fit(x)
    
    ## get centroids ##
    c = model.cluster_centers_
    
    ## gaussian function deviation ## 
    sigma =  ((np.max(c) - np.min(c)) / np.sqrt(2*k))
    
    #plt.plot(c[:,0], c[:, 1], 'ok')
    
    ######## Propagation ##########
    # Create G matrix (The outputs of every neuron for each pattern)  shape: (N. samples of X, k neurons)
    G = np.zeros((x.shape[0], k))
    
    for i in range(x.shape[0]):
        for j in range(k):
            """
                Gaussian function
            """
            distance = np.linalg.norm(x[i, ] - c[j, ], 2)   # norm two (euclidian norm)
            G[i, j] = np.exp((-1/sigma**2) * distance**2)   
    
    ######### Supervised training ########
    # pseudo inverse method
    W = np.dot(np.linalg.pinv(G), y)
    
    # plot dataset
    plotData(x.T, y.T)
    
    ###### make the decision surface ######
    ## get 2-dimention min max
    xmin, ymin = np.min(x[:, 0]) - 0.5, np.min(x[:, 1]) - 0.5
    xmax, ymax = np.max([x[:, 0]]) + 0.5, np.max(x[:, 1 ]) +0.5
    
    ## now we have to create a mesh with that limits (100x100) points
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    
    ## flat xx and yy
    xx = xx.ravel()
    yy = yy.ravel()

    ### Propagation ###
    G = np.zeros((xx.shape[0], k))

    for i in range(xx.shape[0]):
        for j in range(k):
            # calculate gaussian function 
            distance = np.linalg.norm((xx[i], yy[i]) - c[j], 2)
            G[i, j] = np.exp((-1/sigma**2) * distance**2)
            
    # reshape xx and yy to it's original form
    xx = xx.reshape(100,100)
    yy = yy.reshape(100,100)
    
    ##### Prediction ######
    zz = np.dot(G, W)
    
    # same shape as xx and yy
    zz = zz.reshape(xx.shape)
    
    ## title
    plt.title("RBF-NN Classification")
    
    ## axis limits
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    
    ## plot contourf
    plt.contourf(xx, yy, zz)
    plt.grid()
    plt.show()  

        
    