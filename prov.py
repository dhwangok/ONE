#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
from sklearn.neighbors import KDTree
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time
import cPickle as pickle
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# kdt = KDTree(X, leaf_size=30, metric='euclidean')
# print kdt.query(X, k=2, return_distance=True)   

class Unsupervised_Nearest_Neighbors():
	def __init__(self):
		pass
	def load_data(self,fea_path=None):
		start = time.clock()

		# data =  np.loadtxt(path, dtype=np.str, delimiter=',')
		data=pd.read_csv(fea_path, dtype=np.str,header=None)
		# print data
		# print 'data done',time.clock()-start
		start = time.clock()
		# print len(data)
		data=np.array(data)
		# X = data[1:, 7:8]		
		# X = [data[1:, 1],data[1:, 2]]

		X=data[1:, :].astype(np.float)
		min_max_scaler = preprocessing.MinMaxScaler()
		X = min_max_scaler.fit_transform(X)


		# for i in range(len(data)):	
		# 	try:
		# 		print i
		# 		x=data[i, :].astype(np.float)
		# 	except Exception,e:
		# 		print i 
		# 		print data[i, :]
		# 		print e
		# 		return

		return X

	def load_id(self,id_path=None):
		ids=pd.read_csv(id_path, dtype=np.str,header=None)
		ids=np.array(ids)
		ids = ids[1:,:].astype(np.str)
		return ids


	def train(self,X=None,fea_path=None,id_path=None):
		
		if X is None:
			print 'load data...'
			X = self.load_data(fea_path=fea_path)
			ids = self.load_id(id_path=id_path)

		print 'training...'
		kdt = KDTree(X, leaf_size=100, metric='euclidean')
		dists, inds =  kdt.query(X, k=11, return_distance=True)
		print dists, inds
		output=''
		for i in range(len(inds)):
			# print int(ids[i][0]),inds[i],dists[i]
			# print i,ids[i][0],len(ids)
			output+=str(int(ids[i][0]))+';'+','.join(str(v) for v in inds[i])+';'+','.join(str(v) for v in dists[i])+'\n'
			output+=str(ids[i][1])+'\n'
			for ind in range(1,len(inds[i])):
				output+=str(ind)+','+str(ids[inds[i][ind]][1])+'\n'
		# print output
		# print ids
		# print ind
		# print dist
		open("C:/danhua/prov/city/prov_city.csv",'wb').write(output)

		self.dump_model(kdt,'unn_kdtree.model')
		# pickle.dumps(kdt,'unn_kdtree.model')


	def dump_model(self,clf, model):
		fw = open(model, 'wb')
		pickle.dump(clf, fw)
		fw.close()

if __name__=='__main__':

	start = time.clock()

	fea_path="C:/danhua/prov/city/city_data1.csv"
	id_path="C:/danhua/prov/city/city_idname.csv"
	unn=Unsupervised_Nearest_Neighbors()
	unn.train(fea_path=fea_path,id_path=id_path)

	print 'done',time.clock()-start,'s'
