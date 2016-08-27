# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import os

mingw_path = 'C:/Users/Administrator/mingw64/bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']


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
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
# from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
import xgboost as xgb
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
        data = pd.read_csv(path, dtype=np.str, header=None)
        # print data
        # print 'data done',time.clock()-start
        start = time.clock()
        # print len(data)
        data = np.array(data)
        # X = data[1:, 7:8]
        # X = [data[1:, 1],data[1:, 2]]

        # print X
        X = data[0:, :-1].astype(np.float)
        # print X[0]
        # min_max_scaler = preprocessing.MinMaxScaler()
        # X = min_max_scaler.fit_transform(X)
        # X = preprocessing.scale(X)

        # print X

        start = time.clock()

        y = data[0:, -1].astype(np.float)  # *data[0:, -1].astype(np.float)
        y = np.ravel(y)
        # print y
        # print '...load done'

        return X, y

    def checkOneByOne(self, precision=10, X_test=None, y_test=None, model=None):

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

        return (1 - 1. * errorNum / len(y_predict)) * 100

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

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.8, random_state=0)

        est = xgb.XGBRegressor(max_depth=3, n_estimators=800,
                               learning_rate=0.005).fit(X_train, y_train)  #3,300,0.05
        #(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, 
            #subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-2,1e-1],
        #              'C': [1, 10, 100, 1000]},
        #             {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        # clf = GridSearchCV(SVR(C=1), tuned_parameters, cv=5)
        # clf.fit(X_train, y_train)
        # print clf.best_params_

        # print X_train, y_train

        # est = GradientBoostingRegressor(n_estimators=500,
        #                                 learning_rate=0.01,
        #                                 max_depth=3,
        #                                 random_state=1,
        # loss='huber').fit(X_train, y_train)  # 'huber'的效果最好  ‘ls’, ‘lad’,
        # ‘huber’, ‘quantile’

        # feature_importance = est.feature_importances_
        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        # print feature_importance

        # est = Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
        #             normalize=False, random_state=None, solver='auto',
        #             tol=0.001).fit(X_train, y_train)

        # est=ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
        # copy_X=True, tol=0.0001, warm_start=False, positive=False,
        # random_state=None, selection='cyclic').fit(X_train, y_train)
        # est=Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
        # tol=0.0001, warm_start=False, positive=False, random_state=None,
        # selection='cyclic').fit(X_train, y_train)
        # est = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True,
        #                 intercept_scaling=1.0, dual=True, verbose=0, random_state=None,
        #                 max_iter=1000).fit(X_train, y_train)
        # est = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1,
        #           shrinking=True, cache_size=200, verbose=False, max_iter=-1).fit(X_train, y_train)
        #  # ‘linear’, ‘poly’, ‘rbf’,‘sigmoid’, ‘precomputed’
        # est = MLPRegressor(hidden_layer_sizes=(20, ), activation='relu', algorithm='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
        # random_state=None, tol=0.0001, verbose=False, warm_start=False,
        # momentum=0.9, nesterovs_momentum=True, early_stopping=False,
        # validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
        # epsilon=1e-08).fit(X_train, y_train)

        # est=SGDRegressor(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0,
        #              epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False).fit(X_train, y_train)
        # est=KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1).fit(X_train, y_train)

        # est_name='gbdt_1000.model'
        output = open(model_path, 'wb')
        pickle.dump(est, output)
        output.close

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
        pickle.dumps(clf, fw)
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

def getModelsort(feature_id):
    data = pd.read_csv("C:/danhua/carset_ex_1/SVR/model_name.csv")
    result = data.sort_values(['model_price','model_year'], ascending=False)
    # print result
    id_sort = list(result['model_id'])
    # print id_sort
    number = id_sort.index(int(feature_id))
    # print feature_id, number
    return str(number)
    # return inttobinary(number)

def getProvsort(feature_id):
    data=pd.read_csv("C:/danhua/carset_ex_1/SVR/prov.csv")
    prov_sort=list(data['id'])
    # print prov_sort
    number=prov_sort.index(int(feature_id))
    # print feature_id,number
    return str(number)
    # return inttobinary(number)

def getCitysort(feature_id):  
    city=pd.read_csv("C:/danhua/carset_ex_1/SVR/city_new_prov.csv")
    result=city.sort_values(['new_prov','city_id'], ascending=True)
    city_sort=list(result['city_id'])
    # print city_sort
    number=city_sort.index(int(feature_id))
    # print feature_id,number
    return str(number)
    # return cityinttobinary(number)


def tobinary(feature_set, feature_id):
    feature_list = list(feature_set)
    feature_list.sort()
    number = feature_list.index(feature_id)
    # return str(number)
    return inttobinary(number)


def inttobinary(number):

    # originalbinary = bin(number).replace("0b", "")  # bin()  "0b"
    # misslength = 6 - len(originalbinary)  # 6
    originalbinary = str(number)
    misslength = 2 - len(originalbinary)
    fillbinary = "0" * misslength + originalbinary
    # print fillbinary
    finalbinary = ''
    for i in fillbinary:
        finalbinary += i + ','
    # print finalbinary
    # finalbinary=[int(x) for x in list(fillbinary)]
    return finalbinary


def citytobinary(feature_set, feature_id):
    feature_list = list(feature_set)
    feature_list.sort()
    number = feature_list.index(feature_id)
    # return str(number)
    return cityinttobinary(number)


def cityinttobinary(number):
    # originalbinary = bin(number).replace("0b", "")  # bin() "0b"
    # misslength = 9 - len(originalbinary)  # 9
    originalbinary = str(number)
    misslength = 3 - len(originalbinary)
    fillbinary = "0" * misslength + originalbinary
    finalbinary = ''
    for i in fillbinary:
        finalbinary += i + ','
    # print finalbinary
    # finalbinary=[int(x) for x in list(fillbinary)]
    return finalbinary


def splitLine(dt):

    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source = dt[6]
    car_status = dt[7]
    mile_age = dt[8]
    post_time = dt[9][:10]  # 取时间的日期，不取具体时刻
    liter = dt[10]  # 有些数据有单位L
    model_year = dt[11]
    model_price = dt[12]
    gear_type=dt[-1]

    price = dt[5]
# car_status,
    return model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price,gear_type


def splitLine1(dt):

    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source = dt[5]
    car_status = dt[6]
    mile_age = dt[7]
    post_time = dt[8][:10]  # 取时间的日期，不取具体时刻
    liter = dt[9]  # 有些数据有单位L
    model_year = dt[10]
    model_price = dt[11]

    price = dt[-1]
# car_source,car_status,
    return model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price

def getDays(reg_date):
    d1 = datetime.datetime(2016, 8, 9)
    year, month, day = reg_date.split('-')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day

def getTDays(reg_date):
    d1 = datetime.datetime(2016, 8, 9)
    year, month, day = reg_date.split('/')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day


def PreprocessData(featureFile, path):
    train_data=pd.read_csv("C:/danhua/carset_ex_1/carset_ex_1_0810/carset_ex_1_gear_drFalse2008.csv")
    test_data=pd.read_csv("C:/danhua/carset_ex_1/carset_ex_1_new/predict_price_series/predict_price_series_1_gear_drop2008.csv")

    train_data['reg_day']=train_data['reg_date'].apply(lambda x:getDays(x))
    test_data['reg_day']=test_data['reg_date'].apply(lambda x:getTDays(x))
    max_reg_day=max(train_data['reg_day'].unique().max(),test_data['reg_day'].unique().max())

    train_data['post_day']=train_data['post_time'].apply(lambda x:getDays(x))
    test_data['post_day']=test_data['post_time'].apply(lambda x:getTDays(x[:9]))
    max_post_day=max(train_data['post_day'].unique().max(),test_data['post_day'].unique().max())



    f = open(path)
    line = f.readline().strip()  # 读取第一行
    line = f.readline().strip()  # 读取第二行
    data = []
    feature = ''
    d1 = datetime.datetime(2016, 8, 9)

    model_set = set()
    car_source_set = set()
    prov_set = set()
    city_set = set()


    while line:
        # dt=line.split('	')
        dt = line.split(',')
        model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price,gear_type = splitLine(
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
    fea_car_source = ''
    fea_prov_set = ''
    fea_city_set = ''
    fea_modelyear_set = ''
    for mod_id_item in model_set:
        fea_model_id += mod_id_item + '\n'
    open('C:/danhua/carset_ex_1/xgb/id_Set/model_id_set_1.csv',
         'wb').write(fea_model_id)
    # for car_source_item in car_source_set:
    # 	fea_car_source+=car_source_item+'\n'
    # open('C:/danhua/carset_ex_1/SVR/id_Set/car_source_set.csv','wb').write(fea_car_source)
    for prov_id_item in prov_set:
        fea_prov_set += prov_id_item + '\n'
    open('C:/danhua/carset_ex_1/xgb/id_Set/prov_id_set_1.csv',
         'wb').write(fea_prov_set)

    for city_id_item in city_set:
        fea_city_set += city_id_item + '\n'
    open('C:/danhua/carset_ex_1/xgb/id_Set/city_id_set_1.csv',
         'wb').write(fea_city_set)

    # for model_year_item in model_year_set:
    #     fea_modelyear_set+=model_year_item+'\n'
    # open('C:/danhua/carset_ex_1/SVR/id_Set/model_year_set_1.csv',
    #      'wb').write(fea_modelyear_set)

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
        # model_id,car_status,
        model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price,gear_type = splitLine(
            dt)

        if model_year=='2009':
            model_year_type=0
        elif model_year=='2010':
            model_year_type=1
        elif model_year=='2011':
            model_year_type=2
        elif model_year=='2012':
            model_year_type=3
        elif model_year=='2013':
            model_year_type=4
        elif model_year=='2014':
            model_year_type=5
        elif model_year=='2015':
            model_year_type=6
        elif model_year=='2016':
            model_year_type=7
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

        # one=['58', 'dafengche', 'souche']
        # two=['273', '51auto', 'baixing', 'carxoo', 'che300_pro', 'cheyipai', 'cn2che', 'ganji','hx2car', 'iautos', 'kuche', 'sohu', 'taoche', 'taotaocar', 'ttpai', 'ttpai_c2c', 'xcar', 'xici', 'youche', 'youxin', 'youxinpai', 'youyiche', 'zg2sc']
        # three=['carking', 'chemao', 'ygche']
        # four=['che101', 'che168', 'che300','chelaike','cheyitao','jiarenzheng','kx','soucheke']
        # five=['ganjihaoche', 'haoche51', 'jiajiahaoche','renrenche']
        # if car_source in two:
        #     weight=1
        # elif car_source in three:
        #     weight=2
        # elif car_source in four:
        #     weight=3
        # else:
        #     weight=4

        year, month, day = reg_date.split('-')
        d2 = datetime.datetime(int(year), int(month), int(day))
        reg_date_day = (d1 - d2).days

        year, month, day = post_time.split('-')
        # print post_year,post_month,post_day
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
        instance = getModelsort(model_id)+','
        instance += getProvsort(prov_id)+','
        instance += getCitysort(city_id)+','
        # instance = tobinary(model_set, model_id)
        # instance += tobinary(prov_set, prov_id)
        # instance += citytobinary(city_set, city_id)
        # instance += tobinary(model_year_set, model_year)
        # print instance
        # instance +=tobinary(car_source_set,car_source)
        instance+=gear_type+','
        instance += str(model_year_type)+ ','
        instance += model_price + ','
        instance += liter + ','
        # instance+=car_status+','
        # instance += str(reg_date_day) + ','
        # instance += str(post_date_day) + ','
        instance+=str(1.0*reg_date_day/max_reg_day)+','
        
        instance+=str(1.0*post_date_day/max_post_day)+','
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
    # prov/city/train_data_city2.csv'C:/danhua/carset_ex_1/train_data_11.csv'
    # "C:/danhua/carset_ex_1/xgb/carset_ex_1_exact.csv"#'C:/danhua/carset_ex_1/order/exact_data_1_0.3.csv' # prov/city/train_data_city1.csv
    raw_data_path = "C:/danhua/carset_ex_1/carset_ex_1_0810/carset_ex_1_gear_drFalse2008.csv"
    featureFile = 'C:/danhua/carset_ex_1/xgb/fea/fea_carset_1_xgb_int_gear_new.csv'

    PreprocessData(featureFile, raw_data_path)
    ln = Learn()
    ln.load_data(featureFile)
    ln.train(path=featureFile,
             model_path='C:/danhua/carset_ex_1/xgb/model/carset_1_xgb_int_gear_new.pkl')

    print 'done', time.clock() - start
