#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)

class K_means():
	def __init__(self):
		pass
	def load_data(self,fea_path=None):
		start=time.clock()
		data=pd.read_csv(fea_path,dtype=np.str,header=None)
		data=np.array(data)
		# print data[-1,:]
		x=data[:,0].astype(np.str)
		X=data[1:, 1:].astype(np.float)

		# X=data[1:, :].astype(np.float)
		min_max_scaler = preprocessing.MinMaxScaler()
		X = min_max_scaler.fit_transform(X)

		pca = PCA(n_components=3)
		X=pca.fit_transform(X)
		# print X
		
		return X,x

	def load_id(self,id_path=None):
		ids=pd.read_csv(id_path, dtype=np.str,header=None)
		ids=np.array(ids)
		ids = ids[1:,:].astype(np.str)

		return ids


	def train(self,k=None,X=None,fea_path=None,id_path=None):

		if X is None:
			print 'load data...'
			X ,x= self.load_data(fea_path=fea_path)
			ids = self.load_id(id_path=id_path)

		print 'training...'
		clf = KMeans(n_clusters=k, random_state=1)
		s = clf.fit(X)
		# print s
		# print clf.cluster_centers_ #中心
		#clf.labels_每个样本所属的簇
		# print X.shape[0]
		print k,clf.inertia_  ##用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
		output=''
		name_output=''
		for j in range(k):
			clu=[]
			clu_name=''
			id_item=''
			for i in range(0,X.shape[0]):
				if j == clf.labels_[i]: 
					id_item+=str(ids[i][0])+','+str(j+1)+'\n'
					clu_name+=str(ids[i][1].decode('utf-8'))+','
			output+=id_item
			name_output+=clu_name+'\n'
			# print clu,clu_name
		open("C:/danhua/prov/city/city_item10.csv",'wb').write(output)
		open("C:/danhua/prov/city/city_cluster10.csv",'wb').write(name_output)


		fignum=1
		fig = plt.figure(fignum, figsize=(1, 3))
		plt.clf()
		ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

		plt.cla()
		clf.fit(X)
		labels = clf.labels_

		ax.scatter(X[:, 2], X[:, 0], X[:, 1], c=labels.astype(np.float))

		ax.w_xaxis.set_ticklabels([])
		ax.w_yaxis.set_ticklabels([])
		ax.w_zaxis.set_ticklabels([])
		# ax.set_xlabel('city_id')
		plt.show()



		# output=''
		# for i in range(len(inds)):
		# 	# print int(ids[i][0]),inds[i],dists[i]
		# 	# print i,ids[i][0],len(ids)
		# 	output+=str(int(ids[i][0]))+';'+','.join(str(v) for v in inds[i])+';'+','.join(str(v) for v in dists[i])+'\n'
		# 	output+=str(ids[i][1])+'\n'
		# 	for ind in range(1,len(inds[i])):
		# 		output+=str(ind)+','+str(ids[inds[i][ind]][1])+'\n'
		





		# self.dump_model(kmeans_model,'kmeans_model.model')

	def dump_model(self,clf,model):
		fw=open(model,'wb')
		pickle.dump(clf,fw)
		fw.close()




if __name__ == '__main__':
	start=time.clock()
	fea_path="C:/danhua/prov/city/city_data1.csv"
	id_path="C:/danhua/prov/city/city_idname.csv"
	km=K_means()
	km.train(k=10,fea_path=fea_path,id_path=id_path)

	print 'done',time.clock()-start,'s'
