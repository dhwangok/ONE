#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import time
import cPickle as pickle


def training(train,features,target):
	# eta = 0.2
 #    max_depth = 6
 #    subsample = 0.8
 #    colsample_bytree = 0.8

	# params = {
 #        "objective": "binary:logistic",
 #        "booster" : "gbtree",
 #        "eval_metric": "auc",
 #        "eta": eta,
 #        "tree_method": 'exact',
 #        "max_depth": max_depth,
 #        "subsample": subsample,
 #        "colsample_bytree": colsample_bytree,
 #        "silent": 1,
 #        "seed": random_state,
 #    }

    # num_boost_round = 115
    # early_stopping_rounds = 10
    # test_size = 0.1
	
	X_train, X_vali = train_test_split(train, test_size=0.2, random_state=0)
	 
	y_train=X_train[target]
	y_vali=X_vali[target]
	X_train=np.array(X_train[features])
	X_vali=np.array(X_vali[features])
	y_train=np.array(y_train)
	y_vali=np.array(y_vali)

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)

	#

 #    est = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

	#xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None,
	                    #evals_result=None, verbose_eval=True, learning_rates=None, xgb_model=None, callbacks=None)
	model_path = open('D:/first/model.pkl', 'wb')
	pickle.dump(clf,model_path)
	model_path.close

	y_pre=clf.predict(X_vali)
	# print y_pre
	score=roc_auc_score(y_vali, y_pre)
	print 'validation:',score

def run_test(test,features):
	model=load_model('D:/first/model.pkl')
	x_test=test[features]
	x_test=np.array(x_test)
	
	y_pre=model.predict(x_test)
	print y_pre
	result=pd.DataFrame({'people_id':test['people_id'],'outcome':y_pre})
	result.to_csv('D:/first/result.csv',index=False)
	





def load_model(model):  
	fr = open(model,'rb')
	Clf= pickle.load(fr)
	fr.close()
	return Clf

def intersect(a, b):
	return list(set(a) & set(b))

def getFeatures(train,test):
	trainName=list(train.columns)
	testName=list(test.columns)
	output=intersect(trainName,testName)
	output.remove('people_id')
	output.remove('activity_id')
	# print sorted(output)
	return sorted(output)




def read_train_test():
	print 'Read people.csv...'
	people=pd.read_csv('D:/first/people.csv',dtype={'people_id':np.str,'activity_id':np.str,'char_38':np.int32},parse_dates=['date'])
	# print people.columns

	print 'Read train.csv...'
	train=pd.read_csv('D:/first/act_train.csv',dtype={'people_id':np.str,'activity_id':np.str,'outcome':np.int8},parse_dates=['date'])

	print 'Read test.csv...'
	test=pd.read_csv('D:/first/act_test.csv',dtype={'people_id':np.str,'activity_id':np.str,'outcome':np.int8},parse_dates=['date'])

	print 'Process tables...'
	for table in [train,test]:
		table['people_id']=table['people_id'].str.lstrip('ppl_')
		table['activity_category']=table['activity_category'].str.lstrip('type ').astype(np.int32)
		for i in range(1,11):
			table['char_'+str(i)]=table['char_'+str(i)].fillna(value='type 0')
			table['char_'+str(i)]=table['char_'+str(i)].str.lstrip('type ').astype(np.int64)

		table['year']=table['date'].dt.year
		table['month']=table['date'].dt.month
		table['day']=table['date'].dt.day
		table.drop('date',axis=1,inplace=True)
		# print table.columns
	people['year']=people['date'].dt.year
	people['month']=people['date'].dt.month
	people['day']=people['date'].dt.day
	people.drop('date',axis=1,inplace=True)
	people['people_id']=people['people_id'].str.lstrip('ppl_')
	people['group_1']=people['group_1'].str.lstrip('group ')
	for i in range(1,10):
		people['char_'+str(i)]=people['char_'+str(i)].fillna(value='type 0')
		people['char_'+str(i)]=people['char_'+str(i)].str.lstrip('type ').astype(np.int32)
	for i in range(10,38):
		people['char_'+str(i)]=people['char_'+str(i)].astype(int)  #

	print 'Merge...'
	train=pd.merge(train,people,how='left',on='people_id')
	train.fillna(value='0')
	# print train.columns

	test=pd.merge(test,people,how='left',on='people_id')
	test.fillna(value='0')

	features=getFeatures(train,test)

	
	return train,test,features

	# df1=train[(train['report_price']<train['model_price']/2)&(train['reg_year']==2016)]
	# train1=train.drop(df1.index)
    # train1.drop('reg_year',axis=1,inplace=True)
 #    gear_type=test['gear_type']
	# test.drop(labels=['gear_type'], axis=1,inplace = True)
	# test.insert(2, 'gear_type',gear_type)



if __name__ == '__main__':
	train,test,features=read_train_test()
	# training(train,features,'outcome')
	run_test(test,features)
