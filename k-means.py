#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:29:54 2020

@author: michaelantia
"""

import numpy as np
import matplotlib.pyplot as plt


# Create toy data set
mu1 = [39, 50]
mu2 = [28, 58]
mu3 = [27, 40]
cov1 = [[10,0],[0,12]]
cov2 = [[15,0],[0,17]]
cov3 = [[8,0],[0,22]]

z1 = np.ones(50).astype(int)  
z2 = np.ones(50).astype(int) * 2
z3 = np.ones(50).astype(int) * 3
x1, y1 = np.random.multivariate_normal(mu1, cov1, 50).T
x2, y2 = np.random.multivariate_normal(mu2, cov2, 50).T
x3, y3 = np.random.multivariate_normal(mu3, cov3, 50).T
x = np.hstack((x1,x2,x3))
y = np.hstack((y1,y2,y3))
z = np.hstack((z1,z2,z3))

X = np.vstack((x,y,z)).T


# Obtain n number of colors
colors_orig = plt.cm.jet(np.linspace(0,1,3))

# Empty array to hold colors
labels_init_orig = []

# Loop through all observations
for i in range(X.shape[0]):
    ## grab the class (convert to int)
    j = X[i,2].astype(int)
    # Assign the class to an associated color
    labels_init_orig.append(colors_orig[j-1,:])


# Plot
plt.figure()    
plt.scatter(X[:,0], X[:,1], color=labels_init_orig)
plt.scatter(X[:,0], X[:,1], color=labels_init_orig)
plt.savefig("Original Clusters.png")
plt.close()


np.random.shuffle(X)
X_data = X[:,:2]


def labelColorCoding(colors, labels):
    labels_init = []
    # Loop through all observations
    for i in range(labels.shape[0]):
        ## grab the class (convert to int)
        j = labels[i].astype(int)
        # Assign the class to an associated color
        labels_init.append(colors[j,:])
    return labels_init

def plot_and_save_fig(data, title, labels, m_k):
    plt.figure()
    plt.scatter(data[:,0], data[:,1], color=labels)
    plt.scatter(m_k[:,0], m_k[:,1], marker="x", s=20, c="black")
    plt.savefig(title)

# Initialize K random points (indices) for cluster centers
def select_k_random_points(data, k):
    idx = np.random.choice(data.shape[0], size=k)
    return data[idx,:]

# Distance function
def euclideanDistance(data, testPoint):
    return np.sqrt(np.sum((data - testPoint)**2, axis=1))

# def compute nxk distances
def k_distances(data, m_k, func):
    distances = np.zeros((data.shape[0], m_k.shape[0]))
    for i in range(m_k.shape[0]):
        distances[:,i] = func(data, m_k[i,:])
    return distances

# Find smallest distance and return index, expected output n by k 
# This is the cluster assignment
def assignToCluster(distances):
    return np.argmin(distances, axis=1)

# Compute new Cluster Centers    
# args data clucsters
def computeClusterCenter(data, clusters, m_k):
    for i in range(k):
        m_k[i,:] = np.mean(data[clusters == i], axis=0)
    return m_k
    
# Compute Cost J
def costFunction(data, distances, cluster_assignments):
    J = 0
    for i in range(cluster_assignments.shape[0]):
        J += distances[i][cluster_assignments[i]]
    return J

# Convergence function
def converge(costOld, costNew, epsilon=0.0001):
    if np.abs(costOld - costNew) < epsilon:
        return True
    else:
        return False


# Select k
k = 5
colors = plt.cm.jet(np.linspace(0,1,k))
firstRun = True
hasConverged = False
iteration = 0
m_k = select_k_random_points(X_data, k)
oldCost = None


while (hasConverged == False):
    
    iteration += 1
    distances = k_distances(X_data, m_k, euclideanDistance) 
    cluster_assignment = assignToCluster(distances)
    m_k = computeClusterCenter(X_data, cluster_assignment, m_k)
    cost = costFunction(X_data, distances, cluster_assignment)
    print("The cost is: " + str(cost))
    #print(m_k)

    #Plot new data pts and centroids
    labelsColors = labelColorCoding(colors, cluster_assignment)
    title = "Iter - #: " + str(iteration)
    plot_and_save_fig(X_data, title, labelsColors, m_k)
    
    
    #check for convergence
    if not firstRun:
        hasConverged = converge(oldCost, cost)
    oldCost = cost
    
    if firstRun:
        plt.scatter(X_data[:,0], X_data[:,1])
        plt.savefig("Iter - #: " + str(iteration - 1))
        firstRun = False
    #Print iter and cost, display cost?

        
    


## Create Data func
## __main__ kwargs
## Make into classes





