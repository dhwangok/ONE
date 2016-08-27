#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
import numpy as np
import pandas as pd
import random
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt 
import time
# KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)



def load_data(fea_path=None):
	start=time.clock()
	data=pd.read_csv(fea_path,dtype=np.str,header=None)
	data=np.array(data)
	# print X[:,1]
	return data

#计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):	
	return np.sqrt(sum((vecA-vecB)*((vecA-vecB).T)))

#随机生成初始的质心
def randCent(dataSet, k):
	dataSet =load_data(fea_path=fea_path)
	dataSet=dataSet[1:,:].astype(np.float)
	n = dataSet.shape[1]  #dim
	centroids = np.mat(np.zeros((k,n)))  #mat()将数组转化为矩阵
	for j in range(n):
		# print dataSet[:,1]
		minJ = min(dataSet[:,j].astype(np.float))
		
		rangeJ = float(max(np.array(dataSet)[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
	# print centroids
	return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	dataSet =load_data(fea_path=fea_path)
	dataSet=dataSet[1:,:].astype(np.float)
	m = dataSet.shape[0]  #numSamples
	clusterAssment = np.mat(np.zeros((m,2))) #create mat to assign data points to a centroid, also holds SE of each point
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m): #for each data point assign it to the closest centroid
			minDist = np.inf
			minIndex = -1
			for j in range(k):
				# print centroids[j,:],dataSet[i,:]
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment[i,0] != minIndex: 
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		# print centroids
		for cent in range(k):#recalculate centroids
			ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]] #get all the point in this cluster
			centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean 
			# print ptsInClust
	return centroids, clusterAssment

def show(dataSet, k, centroids, clusterAssment):
	dataSet =load_data(fea_path=fea_path)
	dataSet=dataSet[1:,:].astype(np.float)	 
	numSamples, dim = dataSet.shape  
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
	for i in xrange(numSamples):  
		markIndex = int(clusterAssment[i, 0])  
		plt.plot(dataSet[i, 0], dataSet[i, -2], mark[markIndex])
		# print dataSet[i, 0],dataSet[i, 1],markIndex  
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
	for i in range(k):  
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
	plt.show()





if __name__ == '__main__':
	start=time.clock()
	fea_path="C:/danhua/prov/city/city_data.csv"
	# km=K_means()
	# km.train(fea_path=fea_path)
	dataMat =load_data(fea_path=fea_path)
	myCentroids, clustAssing= kMeans(dataMat,5)
	# print myCentroids
	show(dataMat, 5, myCentroids, clustAssing) 

	print 'done',time.clock()-start,'s'
