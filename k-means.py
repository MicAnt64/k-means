#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:29:54 2020

@author: michaelantia
"""

import numpy as np
import matplotlib.pyplot as plt

# Create Dataset in 2D space (100, 100)
# Each sample will have 2 dims (x,y)
# Each class will have 50 data points

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
np.random.shuffle(X)
labels = ['red' if c == 1 else 'green' if c == 2 else 'blue' for c in X[:,2]] 
X_data = X[:,:2]

def plot_and_save_fig():
    plt.scatter(X[:,0], X[:,1], color=labels)
    plt.savefig("OriginalClusters.png")
    plt.show()


# Select k
k = 3

# Initialize K random points (indices) for cluster centers
def select_k_random_points(data, k):
    idx = np.random.choice(data.shape[0], size=k)
    return data[idx,:]

m_k = select_k_random_points(X_data, k)


# Distance function
def euclideanDistance(data, testPoint):
    return np.sqrt(np.sum((data - testPoint)**2, axis=1))

# def compute nxk distances

def k_distances(data, m_k, func):
    distances = np.zeros((data.shape[0], m_k.shape[0]))
    for i in range(m_k.shape[0]):
        distances[:,i] = func(data, m_k[i,:])
    return distances

distances = k_distances(X_data, m_k, euclideanDistance) 

# Find smallest distance and return index, expected output n by k 
# This is the cluster assignment
cluster_assignment = np.argmin(distances, axis=1)

