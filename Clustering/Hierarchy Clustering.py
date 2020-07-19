#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl



dataset = pd.read_csv("Mall_Customers.csv")
dataset.head()
X=dataset.iloc[:,[3,4]].values


# In[30]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sh
dendrogram = sh.dendrogram(sh.linkage(X,method='ward'))
mpl.rc('figure',figsize=(18,18))
# mpl._version_
style.use('ggplot')
plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('E dist')
plt.show()


# In[17]:


model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y=model.fit_predict(X)
y


# In[26]:


mpl.rc('figure',figsize=(9,8))
# mpl._version_
style.use('ggplot')
plt.scatter(X[y==0,0],X[y==0,1],c='red',label='cluster1')
plt.scatter(X[y==1,0],X[y==1,1],c='green',label='cluster2')
plt.scatter(X[y==2,0],X[y==2,1],c='blue',label='cluster3')
plt.scatter(X[y==3,0],X[y==3,1],c='yellow',label='cluster4')
plt.scatter(X[y==4,0],X[y==4,1],c='orange',label='cluster5')
plt.legend()
plt.title("Clustering via Hierarchy")
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(0-100)')
plt.show()

