# -*- coding: UTF-8 -*-
# __author__ = 'maj'

import heapq
import numpy as np
import pandas as pd
import time
import cPickle as pickle
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
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
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


class Learn():

    def __init__(self):
        pass

    def load_data(self, path):
        start = time.clock()
        # print path
        # data =  np.loadtxt(path, dtype=np.str, delimiter=',')
        data = pd.read_csv(path, dtype=np.str)
        # print data
        # print 'data done',time.clock()-start
        start = time.clock()
        # print len(data)
        data = np.array(data)
        # X = data[1:, 7:8]
        # X = [data[1:, 1],data[1:, 2]]
        
        df= pd.DataFrame(data=data, columns=['model_id','city_id','prov_id','model_year','model_price','liter','reg_date_day','post_date_day','mile_age','price'])
        just_dummies = pd.get_dummies(df['prov_id'],drop_first=True)
        print just_dummies



        # print X
        X = data[0:, :-1].astype(np.float)
        # print X[0]

        start = time.clock()

        y = data[0:, -1].astype(np.float)  # *data[0:, -1].astype(np.float)
        y = np.ravel(y)
        # print y
        # print '...load done'
        return X, y

    def checkOneByOne(self, precision=5, X_test=None, y_test=None, model=None):

        errorNum = 0
        y_predict = model.predict(X_test)
        for i in range(0, len(y_predict)):
            # print y_test[i],y_predict[i]
            if abs(y_test[i] - y_predict[i]) > y_test[i] / 100 * precision:
                errorNum += 1
        print
        print 'errorNum:', errorNum,
        print 'allNum:', len(y_predict),
        print 'precision:', (1 - 1. * errorNum / len(y_predict)) * 100
        print

    def sampleInstance(self, X, y, sampleRate):
        indexs = [i for i in range(len(y))]
        # print sampleRate*len(y)
        # print indexs
        trainIndexs = random.sample(indexs, int(sampleRate * len(y)))
        testIndexs = []
        for i in range(len(y)):
            if i not in trainIndexs:
                testIndexs.append(i)

        X_train = X[trainIndexs, :]
        y_train = y[trainIndexs]

        X_test = X[testIndexs, :]
        y_test = y[testIndexs]

        return X_train, y_train, X_test, y_test

    def train(self, path=None, model_path=None, X=None, y=None, clf=None, algorithm='LogisticRegression'):
        # print 'train...'
        # print 'load data'
        if X is None:
            X, y = self.load_data(path)

        print 'instance num:', len(X)

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
        X_train, y_train, X_test, y_test = self.sampleInstance(
            X, y, sampleRate)

        est = GradientBoostingRegressor(n_estimators=500,
                                        learning_rate=0.01,
                                        max_depth=3,
                                        random_state=1,
                                        loss='huber').fit(X_train, y_train)  # 'huber'的效果最好

        # feature_importance = est.feature_importances_
        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        # print feature_importance
        # est=ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
        # copy_X=True, tol=0.0001, warm_start=False, positive=False,
        # random_state=None, selection='cyclic').fit(X_train, y_train)

        # est=Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
        # tol=0.0001, warm_start=False, positive=False, random_state=None,
        # selection='cyclic').fit(X_train, y_train)

        # est=RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        #                       max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=1, verbose=0, warm_start=False).fit(X_train,y_train)

        # est=Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
        # normalize=False, random_state=None, solver='auto',
        # tol=0.001).fit(X_train,y_train)

        # est=KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None).fit(X_train, y_train)

        # est_name='gbdt_1000.model'
        self.dump_model(est, model_path)

        # print 'check result:'

        self.checkOneByOne(precision=10,
                           X_test=X_test,
                           y_test=y_test,
                           model=est)

        # loo = cross_validation.KFold(10)
        # scores = cross_validation.cross_val_score(est, X, y, scoring='mean_squared_error', cv=loo,)
        # print(scores.mean())

        # mse=mean_squared_error(y, est.predict(X))
        # print 'mse:',mse

    def dump_model(self, clf, model):
        fw = open(model, 'w')
        pickle.dump(clf, fw)
        fw.close()

    def load_model(self, model):
        fr = open(model)
        return pickle.load(fr)

    # def check(self,y, pre_topk):
    # 	hit = 0.0
    # 	for pt in pre_topk:
    # 		if y[pt[0]]==1:
    # 			hit += 1
    # 	return hit


def findIndex(feature_id, feature_set):
    index = -1
    for _id in feature_set:
        index += 1
        if _id == feature_id:
            return index


def oneHotEncoding(instance, feature_set, feature_id):
    feature_index = findIndex(feature_id, feature_set)
    for i in range(len(feature_set)):
        if i == feature_index:
            instance += '1' + ','
        else:
            instance += '0' + ','
    return instance
# 对离散的数据二制化


def splitLine(dt):

    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source = dt[6]
    # car_status=dt[7]
    mile_age = dt[8]
    post_time = dt[9][:10]  # 取时间的日期，不取具体时刻
    liter = dt[10][:-1]  # 有些数据有单位L
    model_year = dt[11]
    model_price = dt[12]

    price = dt[5]
# car_source,car_status,
    return model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price


def splitLine1(dt):

    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4][:10]      
    car_source = dt[5]
    # car_status=dt[6]
    mile_age = dt[7]
    post_time = dt[8][:10]  # 取时间的日期，不取具体时刻
    liter = dt[9][:-1]  # 有些数据有单位L
    model_year = dt[10]
    model_price = dt[11]

    price = dt[-1]
# car_source,car_status,
    return model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price

def splitLine2(dt):

    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source = dt[5]
    # car_status=dt[7]
    mile_age = dt[7]
    post_time = dt[8]  # 取时间的日期，不取具体时刻
    liter = dt[9][:-1]  # 有些数据有单位L
    model_year = dt[10]
    model_price = dt[11]

    price = dt[12]
# car_source,car_status,
    return model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price


def PreprocessData(featureFile, path):

    f = open(path)
    line = f.readline().strip()  # 读取第一行
    line = f.readline().strip()  # 读取第二行
    data = []
    feature = ''
    d1 = datetime.datetime(2016, 6, 28)

    model_set = set()
    car_source_set = set()
    prov_set = set()
    city_set = set()

    while line:
        # dt=line.split('	')
        dt = line.split(',')
        model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)

        model_set.add(model_id)
        # car_source_set.add(car_source)
        prov_set.add(prov_id)
        city_set.add(city_id)

        line = f.readline().strip()
    print len(model_set), len(prov_set), len(city_set)

    # print len(model_set),len(car_source_set),len(prov_set),len(city_set)

    # 将model_set,carsource_set,prov_set,city_set写入csv
    fea_model_id = ''
    # fea_car_source = ''
    fea_prov_set = ''
    fea_city_set = ''
    for mod_id_item in model_set:
        fea_model_id += mod_id_item + '\n'
    open('C:/danhua/carset_ex_1/id_Set/model_id_set_1_11_1.csv',
         'wb').write(fea_model_id)
    

    # for car_source_item in car_source_set:
    # 	fea_car_source+=car_source_item+'\n'
    # open('car_source_set.csv','wb').write(fea_car_source)
    for prov_id_item in prov_set:
        fea_prov_set += prov_id_item + '\n'
    open('C:/danhua/carset_ex_1/id_Set/prov_id_set_1_11_1.csv',
         'wb').write(fea_prov_set)
   

    for city_id_item in city_set:
        fea_city_set += city_id_item + '\n'
    open('C:/danhua/carset_ex_1/id_Set/city_id_set_1_11_1.csv',
         'wb').write(fea_city_set)


    # for model in model_set:
    # 	print model
    # return

    # webSourceArray=['taoche', 'iautos', 'ygche', 'haoche51', 'chemao', 'cn2che', 'taotaocar', 'renrenche', 'ganji', 'che168', 'cheyitao', 'hx2car', '51auto', 'soucheke', 'ganjihaoche', 'youxin', 'che101', '58', 'youche', 'souche', 'sohu', 'xcar', 'zg2sc']

    f = open(path)
    line = f.readline().strip()
    line = f.readline().strip()
    k = 0
    while line:
        k += 1
        # print line
        dt = line.split(',')
        # dt=line.split('	')
        # model_id,car_source,car_status,
        model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)
        # print model_price
        # reg_year=datetime.datetime(int(year))
        # not_good=['58','dafengche','souche']

        # if city_id not in ['44']:
        # 	line=f.readline().strip()
        # 	continue

        # cs=['che168','ganjihaoche','renrenche','haoche51']
        # if car_source not in cs:
        # 	line=f.readline().strip()
        # 	continue

        year, month, day = reg_date.split('-')
        d2 = datetime.datetime(int(year), int(month), int(day))
        reg_date_day = (d1 - d2).days

        year, month, day = post_time.split('-')
        d2 = datetime.datetime(int(year), int(month), int(day))
        post_date_day = (d1 - d2).days

        # if float(price) < float(model_price)*(0.9**(1.*reg_date_day/365))*0.8 or float(price) > min(float(model_price)*(0.9**(1.*reg_date_day/365))*1.2,float(model_price)):
        # 	line=f.readline().strip()
        # 	continue

        instance = ''
        # one hot encoing model_id
        # instance = oneHotEncoding(instance, model_set, model_id)
        # instance=oneHotEncoding(instance,car_source_set,car_source)
        # instance = oneHotEncoding(instance, prov_set, prov_id)
        # instance = oneHotEncoding(instance, city_set, city_id)
        # print instance
        instance+=model_id+','
        instance+=city_id+','
        instance+=prov_id+','
        instance += model_year + ','
        instance += model_price + ','
        instance += liter + ','
        # instance+=car_status+','
        instance += str(reg_date_day) + ','
        instance += str(post_date_day) + ','
        instance += mile_age + ','

        instance += price + '\n'

        # feature+=dt[2]+','+dt[3]+','+dt[4]+','+dt[5]+','+str(d)+','+str(index)+','+dt[9]+','+dt[10]+','+dt[7]+'\n'
        # webSource.add(dt[8])
        # print d
        # 28 23 31 344
        # 27 20 31 248

        # print len(instance[:-1].split(','))

        if k == 1:
            print instance
            print len(instance.split(','))
        feature += instance

        line = f.readline().strip()

    # print webSource

    # print feature
    open(featureFile, 'wb').write(feature)


if __name__ == '__main__':

    start = time.clock()
    raw_data_path = "C:/danhua/carset_ex_1/train_data_12.csv"
    # raw_data_path ='C:/danhua/carset/analysis/0.25/exact_data_0.25/exact_data_1_11_0.2.csv'
    featureFile = 'C:/danhua/carset_ex_1/fea/fea_carset_1_12_gbrt_1.csv'
    


    PreprocessData(featureFile, raw_data_path)
    ln = Learn()
    ln.load_data(featureFile)
    ln.train(path=featureFile,
             model_path='C:/danhua/carset_ex_1/model/carset_1_12_gbrt_1.model')
   


    print 'done', time.clock() - start
