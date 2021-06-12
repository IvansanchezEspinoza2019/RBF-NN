# Libraries
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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