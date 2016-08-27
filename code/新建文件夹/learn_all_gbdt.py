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
from sklearn import svm
from sklearn import cross_validation
import datetime
import random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class Learn():
 
	def __init__(self,series_id=None,n_estimators=None,learning_rate=None,max_depth=None):
		self.series_id=series_id
		self.n_estimators=n_estimators
		self.learning_rate=learning_rate
		self.max_depth=max_depth
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
			# print y_test[i],y_predict[i]
			if abs(y_test[i]-y_predict[i])>y_test[i]/100*precision:
				errorNum+=1
		print 'series_id:',self.series_id,
		print 'n_estimators:',self.n_estimators,
		print 'learning_rate:',self.learning_rate,
		print 'max_depth:',self.max_depth,
		print 'errorNum:',errorNum,
		print 'allNum:',len(y_predict),
		print 'precision:',(1-1.*errorNum/len(y_predict))*100
		

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
				

	def train(self,path=None, model_path=None, sample_rate=0.8,n_estimators=100,X=None,y=None,clf=None,algorithm='LogisticRegression'):
		# print 'train...'
		# print 'load data'
		if X is None:
			X,y = self.load_data(path)

		# print 'instance num:',len(X)

		# print 'training...'
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

		X_train,y_train,X_test,y_test=self.sampleInstance(X,y,sample_rate)

		est = GradientBoostingRegressor(n_estimators=n_estimators,\
			learning_rate=0.01,\
			max_depth=3,\
			random_state=1,\
			loss='huber').fit(X_train, y_train)

		# est=Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
		# 	normalize=False, random_state=None, solver='auto', tol=0.001)

		# est_name='gbdt_1000.model'
		self.dump_model(est,model_path)

		# print 'check result:'

		self.checkOneByOne(precision=10,\
							X_test=X_test,\
							y_test=y_test,\
							model=est)

		# loo = cross_validation.KFold(10)
		# scores = cross_validation.cross_val_score(est, X, y, scoring='mean_squared_error', cv=loo,)
		# print(scores.mean())

		# mse=mean_squared_error(y, est.predict(X)) 
		# print 'mse:',mse   


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

def oneHot_city_dis(instance,city_id,city_set,city_dis_map):
	
	for city in city_set:
		if city_id in city_dis_map.keys():
			if city in city_dis_map[city_id].keys():
				instance+=str(city_dis_map[city_id][city])+','
				# print city_id,city,city_dis_map[city_id][city]
			else:
				print city_id,city,'not find 11'
				# print 'city 2: %s in (city 1: %s,city 2: %s) not find'%(city,city_id,city)
		else:
			print city_id,city,'not find 00'
			# print 'city 1: %s in (city 1: %s,city 2: %s) not find'%(city_id,city_id,city)

	return instance

def splitLine(dt):

	model_id=dt[1]
	prov_id=dt[2]
	city_id=dt[3]
	reg_date=dt[4]	
	car_source=dt[5]
	car_status=dt[6]
	mile_age=dt[7]
	post_time=dt[8][:10]
	liter =dt[9]
	model_year=dt[10]
	model_price=dt[11]
	
	price =dt[12]

	return model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_year,model_price,price
	

def check_raw_data(raw_data_path):

	try:
		f = open(raw_data_path)
	except:
		return False,-1
	line=f.readline().strip()	
	data_num=0
	while line:
		data_num+=1
		line=f.readline().strip()

	data_num-=1
	if data_num<1000:
		return False,data_num
	else:
		return True,data_num



def PreprocessData(featureFile,path,citys_dis_path):

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

	city_dis_map={}

	# citys_dis_file = open(citys_dis_path)
	# citys_dis_line=citys_dis_file.readline().strip()
	# while citys_dis_line:
	# 	city_1,city_2,dis=citys_dis_line.split(',')
	# 	if city_1 not in city_dis_map.keys():
	# 		# print 'city dis error'
	# 	# else:
	# 		city_dis_map[city_1]={}
	# 		city_dis_map[city_1][city_2]=float(dis)
	# 	else:
	# 		city_dis_map[city_1][city_2]=float(dis)

	# 	citys_dis_line=citys_dis_file.readline().strip()

	data_num=0
	while line:
		data_num+=1

		dt=line.split(',')
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


	 
	# print len(model_set),len(car_source_set),len(prov_set),len(city_set)
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
		dt=line.split(',')
		model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_year,model_price,price=splitLine(dt)


		year,month,day=reg_date.split('-')
		d2 = datetime.datetime(int(year),int(month),int(day))
		reg_date_day=(d1-d2).days

		year,month,day=post_time.split('-')
		d2 = datetime.datetime(int(year),int(month),int(day))	
		post_date_day=(d1-d2).days	


		instance=''
		# one hot encoing model_id
		# instance=oneHotEncoding(instance,model_set,model_id)
		# instance=oneHotEncoding(instance,car_source_set,car_source)
		instance=oneHotEncoding(instance,prov_set,prov_id)
		instance=oneHotEncoding(instance,city_set,city_id)	

		# instance=oneHot_city_dis(instance,city_id,city_set,city_dis_map)	
		# print instance
		instance+=model_year+','
		instance+=model_price+','
		instance+=liter+','
		# instance+=car_status+','
		instance+=str(reg_date_day)+','
		instance+=str(post_date_day)+','
		instance+=mile_age+','

		instance+=price+'\n'

		# print d
		# 28 23 31 344
		# 27 20 31 248

		# print len(instance[:-1].split(','))
		
		if k==1:
			# print instance
			# print len(instance.split(','))
			pass
		feature+=instance


		line=f.readline().strip()

	# print webSource
	
	# print feature
	open(featureFile, 'wb').write(feature)

def ckeck_raw_data(series_id,path):

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

	city_dis_map={}

	model_count_dic={}


	data_num=0
	while line:
		data_num+=1

		dt=line.split('	')
		model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_year,model_price,price=splitLine(dt)
		# cs=['che168','ganjihaoche','renrenche','haoche51']
		# if dt[8] not in cs:
		# 	line=f.readline().strip()
		# 	continue
		if model_id in model_count_dic.keys():
			model_count_dic[model_id]+=1
		else:
			model_count_dic[model_id]=1

		# model_set.add(model_id)
		# car_source_set.add(car_source)
		# prov_set.add(prov_id)
		# city_set.add(city_id)
		# print d

		line=f.readline().strip()

	# for k,v in model_count.items():
	# 	print k,v
	model_count=sorted(model_count_dic.iteritems(), key=lambda d:d[1], reverse = True ) 
	
	
	hot_model_count=0
	hot_model_num=0
	sum_model_count=0
	for kv in model_count:
		print kv[0],kv[1]
		sum_model_count+=kv[1]
		if kv[1]>100:
			hot_model_num+=1
			hot_model_count+=kv[1]

	print series_id,'======================',sum_model_count,len(model_count),hot_model_num,1.*hot_model_count/sum_model_count


def make_model_and_validation(series_id=1,n_estimators=100,sample_rate=0.8,feature_path=None,model_path=None,learning_rate=None,max_depth=None):


	ln=Learn(series_id=series_id,\
		n_estimators=n_estimators,\
		learning_rate=learning_rate,\
		max_depth=max_depth)
	ln.load_data(feature_path)
	ln.train(path=feature_path,model_path=model_path,n_estimators=n_estimators,sample_rate=sample_rate)

	# print 'series_id:',series_id,'n_estimators:',n_estimators	

def gener_feature(series_id=1,citys_dis_path=None,feature_path=None,raw_data_path=None):

	PreprocessData(feature_path,raw_data_path,citys_dis_path)

def get_file_path(series_id,n_estimators):
	raw_data_path='../data/exact_train_data/exact_data_%s.csv'%series_id
	feature_path='../data/feature_exact_train_raw/fea_raw_exact_train_series_%s.csv'%series_id
	model_path='../data/model_gbdt/model_raw_exact_train_series_%s_%s.model'%(series_id,n_estimators)
	citys_dis_path='../data/dis_features.csv'	

	return raw_data_path,feature_path,model_path,citys_dis_path

if __name__=='__main__':

	start = time.clock()

	series_id=1

	n_estimators=500
	learning_rate=0.01
	max_depth=3

	sample_rate=0.8
	random_seed=1

	random.seed(random_seed)
	start_series_id=1
	end_series_id=2485

	for series_id in range(start_series_id,end_series_id+1):
		print series_id,'===================='
		
		raw_data_path,feature_path,model_path,citys_dis_path=get_file_path(series_id,n_estimators)
		
		isOk,error_num=check_raw_data(raw_data_path)
		print series_id,n_estimators,isOk,error_num
		if error_num>=1000:

			# ckeck_raw_data(series_id,raw_data_path)
			# continue

			gener_feature(series_id=series_id,\
								citys_dis_path=citys_dis_path,\
								raw_data_path=raw_data_path,\
								feature_path=feature_path)

			make_model_and_validation(series_id=series_id,\
								n_estimators=n_estimators,\
								learning_rate=learning_rate,\
								max_depth=max_depth,\
								feature_path=feature_path,\
								model_path=model_path)	
		else:
			if error_num==-1:
				print raw_data_path,'not find'
			else:
				print raw_data_path,'data num',error_num,'<1000'



	print 'done',time.clock()-start,'s'


