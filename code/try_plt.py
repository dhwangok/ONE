# -*- coding: UTF-8 -*-   
# __author__ = 'maj'

import heapq
import numpy as np
import pandas as pd
import time
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation
import datetime
import random
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

class Learn():
 
	def __init__(self):
		pass

	def load_data(self,path):
		start = time.clock()
		# print path
		# data =  np.loadtxt(path, dtype=np.str, delimiter=',')
		data=pd.read_csv(path, dtype=np.str)
		# print data
		# print 'data done',time.clock()-start
		start = time.clock()
		# print len(data)
		data=np.array(data)
		# X = data[1:, 7:8]		
		# X = [data[1:, 1],data[1:, 2]]

		# print X
		X = data[0:, :-1].astype(np.float)
		# print X[0]

		start = time.clock()

		y = data[0:, -1].astype(np.float)#*data[0:, -1].astype(np.float)
		y = np.ravel(y)
		# print y
		# print '...load done'
		return X,y

	def checkOneByOne(self,precision=5,X_test=None,y_test=None,model=None):
		errorNum=0
		y_predict=model.predict(X_test)
		for i in range(0,len(y_predict)):
			print y_test[i],y_predict[i]
			if abs(y_test[i]-y_predict[i])>y_test[i]/100*precision:
				errorNum+=1
		print 
		print 'errorNum:',errorNum,
		print 'allNum:',len(y_predict),
		print 'precision:',(1-1.*errorNum/len(y_predict))*100
		print
	def sampleInstance(self,X,y,sampleRate):
		indexs=[i for i in range(len(y))]
		# print sampleRate*len(y)
		# print indexs
		trainIndexs=random.sample(indexs,int(sampleRate*len(y)))
		testIndexs=[]
		for i in range(len(y)):
			if i not in trainIndexs:
				testIndexs.append(i)

		X_train=X[trainIndexs,:]
		y_train=y[trainIndexs]

		X_test=X[testIndexs,:]
		y_test=y[testIndexs]

		return X_train,y_train,X_test,y_test


	def train(self,path=None, model_path=None, X=None,y=None,clf=None,algorithm='LogisticRegression'):
		# print 'train...'
		# print 'load data'
		if X is None:
			X,y = self.load_data(path)

		print 'instance num:',len(X)

		print 'training...'
		# return
		# ln=int(len(X)*0.8)
		# ln=int(400*0.8)


		# X_train=X[:ln,:]
		# X_test=X[ln+1:,:]
		# y_train=y[:ln]
		# y_test=y[ln+1:]


		# ln=int(len(X)*0.2)
		# X_test=X[:ln,:]
		# X_train=X[ln+1:,:]
		# y_test=y[:ln]
		# y_train=y[ln+1:]
		sampleRate = 0.8
		X_train,y_train,X_test,y_test = self.sampleInstance(X,y,sampleRate)

		est = GradientBoostingRegressor(n_estimators=1000,\
			learning_rate=0.005,\
			max_depth=3,\
			random_state=1,\
			loss='huber').fit(X_train, y_train)


		# est=Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
		# 	normalize=False, random_state=None, solver='auto', tol=0.001)

		# est_name='gbdt_1000.model'
		self.dump_model(est,model_path)

		print 'check result:'

		# self.checkOneByOne(precision=10,\
		# 					X_test=X_test,\
		# 					y_test=y_test,\
		# 					model=est)

		# loo = cross_validation.KFold(10)
		# scores = cross_validation.cross_val_score(est, X, y, scoring='mean_squared_error', cv=loo,)
		# print(scores.mean())

		# mse=mean_squared_error(y, est.predict(X)) 
		# print 'mse:',mse   
	def plt22(self,path=None):
		print u"开始画图了"
		param_range=np.linspace(.1, 1.0, 10)
		X,y=self.load_data(path)
		est = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.005,max_depth=3,random_state=1,loss='huber')
		# est = linear_model.Ridge (alpha = .5)
		train_sizes, train_scores, test_scores= learning_curve(est,X,y,train_sizes=param_range)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.title("Validation Curve with GBR")
		plt.xlabel("$n_estimators$")
		plt.ylabel("Score")
		plt.ylim(0.0, 1.1)
		plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
		plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2, color="r")
		plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
		plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
		plt.legend(loc="best")
		plt.show()



	def dump_model(self,clf, model):
		fw = open(model, 'w')
		pickle.dump(clf, fw)
		fw.close()

	def load_model(self,model):
		fr = open(model)
		return pickle.load(fr)

	def check(self,y, pre_topk):
		hit = 0.0
		for pt in pre_topk:
			if y[pt[0]]==1:
				hit += 1
		return hit

def findIndex(feature_id,feature_set):
	index=-1
	for _id in feature_set:
		index+=1
		if _id==feature_id:
			return index
def oneHotEncoding(instance,feature_set,feature_id):
	feature_index=findIndex(feature_id,feature_set)
	for i in range(len(feature_set)):
		if i==feature_index:
			instance+='1'+','
		else:
			instance+='0'+','
	return instance

def splitLine(dt):

	model_id=dt[1]
	prov_id=dt[2]
	city_id=dt[3]
	reg_date=dt[4]	
	car_source=dt[6]
	car_status=dt[7]
	mile_age=dt[8]
	post_time=dt[9][:10]
	liter =dt[10]
	model_year=dt[11]
	model_price=dt[12]
	
	price =dt[5]

	return model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_year,model_price,price

def PreprocessData(featureFile,path):


	f = open(path)
	line=f.readline().strip()	
	line=f.readline().strip()
	data=[]
	feature=''
	d1 = datetime.datetime(2016, 3, 3)

	model_set=set()
	car_source_set=set()
	prov_set=set()
	city_set=set()

	while line:
		dt=line.split('	')
		model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_year,model_price,price=splitLine(dt)
		# cs=['che168','ganjihaoche','renrenche','haoche51']
		# if dt[8] not in cs:
		# 	line=f.readline().strip()
		# 	continue

		model_set.add(model_id)
		car_source_set.add(car_source)
		prov_set.add(prov_id)
		city_set.add(city_id)
		# print d

		line=f.readline().strip()


	print len(model_set),len(car_source_set),len(prov_set),len(city_set)

	#将model_set,carsource_set,prov_set,city_set写入csv
	fea_model_id = ''
	fea_car_source = ''
	fea_prov_set = ''
	fea_city_set = ''
	model_id_set_path = 'C:/danhua//model_id_set_3.csv'
	car_source_set_path = 'C:/danhua/car_source_set_3.csv'
	prov_set_path = 'C:/danhua/prov_set_3.csv'
	city_set_path = 'C:/danhua/city_set_3.csv'

	for mod_id_item in model_set:
		fea_model_id+=mod_id_item+'\n'
	open(model_id_set_path,'wb').write(fea_model_id)
	for car_source_item in car_source_set:
		fea_car_source+=car_source_item+'\n'
	open(car_source_set_path,'wb').write(fea_car_source)
	for prov_id_item in prov_set:
		fea_prov_set+=prov_id_item+'\n'
	open(prov_set_path,'wb').write(fea_prov_set)

	for city_id_item in city_set:
		fea_city_set+=city_id_item+'\n'
	open(city_set_path,'wb').write(fea_city_set)

	# for model in model_set:
	# 	print model
	# return

	# webSourceArray=['taoche', 'iautos', 'ygche', 'haoche51', 'chemao', 'cn2che', 'taotaocar', 'renrenche', 'ganji', 'che168', 'cheyitao', 'hx2car', '51auto', 'soucheke', 'ganjihaoche', 'youxin', 'che101', '58', 'youche', 'souche', 'sohu', 'xcar', 'zg2sc']

	
	f = open(path)
	line=f.readline().strip()
	line=f.readline().strip()	
	k=0
	while line:
		k+=1
		dt=line.split('	')
		model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_year,model_price,price=splitLine(dt)


		
		# if city_id not in ['44']:
		# 	line=f.readline().strip()
		# 	continue

		# cs=['che168','ganjihaoche','renrenche','haoche51']
		# if car_source not in cs:
		# 	line=f.readline().strip()
		# 	continue


		year,month,day=reg_date.split('-')
		d2 = datetime.datetime(int(year),int(month),int(day))
		reg_date_day=(d1-d2).days

		year,month,day=post_time.split('-')
		d2 = datetime.datetime(int(year),int(month),int(day))	
		post_date_day=(d1-d2).days	
		# data.append(dt)
		# if float(price) < float(model_price)*(0.9**(1.*reg_date_day/365))*0.8 or float(price) > float(model_price)*(0.9**(1.*reg_date_day/365))*1.2:
		# 	line=f.readline().strip()
		# 	continue

		instance=''
		# one hot encoing model_id
		instance=oneHotEncoding(instance,model_set,model_id)
		# instance=oneHotEncoding(instance,car_source_set,car_source)
		instance=oneHotEncoding(instance,prov_set,prov_id)
		instance=oneHotEncoding(instance,city_set,city_id)		
		# print instance
		instance+=model_year+','
		instance+=model_price+','
		instance+=liter+','
		# instance+=car_status+','
		instance+=str(reg_date_day)+','
		instance+=str(post_date_day)+','
		instance+=mile_age+','

		instance+=price+'\n'


			

		# feature+=dt[2]+','+dt[3]+','+dt[4]+','+dt[5]+','+str(d)+','+str(index)+','+dt[9]+','+dt[10]+','+dt[7]+'\n'
		# webSource.add(dt[8])
		# print d
		# 28 23 31 344
		# 27 20 31 248

		# print len(instance[:-1].split(','))
		
		# if k==1:
		# 	print instance
		# 	print len(instance.split(','))
		feature+=instance

		line=f.readline().strip()

	# print webSource
	
	# print feature
	open(featureFile, 'wb').write(feature)


if __name__=='__main__':

	start = time.clock()
	raw_data_path='C:/danhua/da_carset/da_car_series_3.csv'
	featureFile='C:/danhua/da_carset_features/fea_da_car_series_3.csv'
	model_path='C:/danhua/analysise/17/da_car_series1_model_3.model'

	# PreprocessData(featureFile,raw_data_path)
	ln=Learn()
	ln.load_data(featureFile)
	# ln.train(path=featureFile,model_path = model_path )
	ln.plt22(path=featureFile)

	print 'done',time.clock()-start


