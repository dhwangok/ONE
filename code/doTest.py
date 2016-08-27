# -*- coding: UTF-8 -*-   
# __author__ = 'lz'

import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn import preprocessing
import sys
import logging
import os

mingw_path = 'C:/Users/Administrator/mingw64/bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']




class aoTest():
    def __init__(self):
        pass


    def load_data(self,path):
        data = pd.read_csv(path,dtype=np.str,header=None)
        data = np.array(data)
        # X = data[0:,:-1].astype(float)
        X = data[0:, :-1].astype(np.float)
        # min_max_scaler = preprocessing.MinMaxScaler()
        # X = min_max_scaler.fit_transform(X)
        # X = preprocessing.scale(X)
        
        y = data[0:,-1].astype(float)
        y = np.ravel(y)
        return X,y

    def load_model(self,model):  
        fr = open(model,'rb')
        Clf= pickle.load(fr)
        fr.close()
        return Clf

    def checkOneByOne(self,precision=10,X_test = None,y_test= None,model= None):

        print'checking...'
        erroNum = 0

        y_predict=model.predict(X_test)*0.95
        instance = ''
        for i in range(0,len(y_predict)):
            # instance+=str(y_test[i])+'  '+str(y_predict[i])+'\n'
            # print  y[i],y_predict[i]
            if abs(y_test[i]-y_predict[i])>y_test[i]/100*precision:
                erroNum+=1
                # print  y[i],y_predict[i]
        
        # open('predict_price.csv','wb').write(instance)
        # print 
        print 'errorNum',erroNum,
        print 'allNum',len(y_predict)
        print 'precision:',(1-1.*erroNum/len(y_predict))*100
        allNum=len(y_predict)
        precision=(1-1.*erroNum/len(y_predict))*100
        


        
        # open('predict_price.csv','wb').write(instance)
        # print j
        # print 'errorNum',erroNum,
        # print 'allNum',len(y_predict)
        # print 'precision:',(1-1.*erroNum/len(y_predict))*100
        # allNum=len(y_predict)
        # precision=(1-1.*erroNum/len(y_predict))*100
        # return erroNum,allNum,precision

        # df=pd.read_csv( "C:/danhua/carset_ex_1/test_data_1_noCity_id29.csv")
        df=pd.read_csv("C:/danhua/carset_ex_1/carset_ex_1_new/predict_price_series/predict_price_series_1_gear_drop2008.csv")  #'C:/danhua/prov/city/test_data_2_noCity_id29.csv'
        df["y_predict"]=y_predict
        df.to_csv('C:/danhua/carset_ex_1/xgb/test1/test_data_1_Predict_1_xgb_int_gear_new_3_300_.05_price_year.csv',index=False) 
        return erroNum,allNum,precision



if __name__=='__main__':
        ln = aoTest()
        X,y = ln.load_data('C:/danhua/carset_ex_1/xgb/fea/fea_test_1_xgb_int_gear_new.csv')

        model=  ln.load_model('C:/danhua/carset_ex_1/xgb/model/carset_1_xgb_int_gear_new.pkl') 
        # print model              
        ln.checkOneByOne(precision=10,X_test=X,y_test =y,model=model)

       
