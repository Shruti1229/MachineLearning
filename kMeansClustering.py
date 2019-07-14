import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets 


iris = datasets.load_iris() 

X = iris.data 
y = iris.target

df = pd.DataFrame(X) 
#print(df.head())
X = scale(df)
#print(X[:5])
kmeans = KMeans(n_clusters = 3).fit(X)
mu = kmeans.cluster_centers_

def plot_cluster_data(X,mu = None):

	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(1,1,1)
	ax.plot(X[:,0],X[:,1],'o')
	if not mu is None:
		ax.plot(mu[0,0],mu[0,1],'o',markerfacecolor = 'red',markersize = 12)
		ax.plot(mu[1,0],mu[1,1],'o',markerfacecolor = 'green',markersize = 12)
		ax.plot(mu[2,0],mu[2,1],'o',markerfacecolor = 'yellow',markersize = 12)
	plt.show()

mu = kmeans.cluster_centers_

plot_cluster_data(X, mu = mu)
