# -*- coding: UTF-8 -*-   
# __author__ = 'lz'

import numpy as np
import pandas as pd
import cPickle as pickle
import sys
import logging



class aoTest():
    def __init__(self):
        pass


    def load_data(self,path):
        data = pd.read_csv(path,dtype=np.str,header=None)
        data = np.array(data)
        # X = data[0:,:-1].astype(float)
        X = data[0:,:-1].astype(float)
        y = data[0:,-1].astype(float)
        y = np.ravel(y)
        return X,y

    def load_model(self,model):
        fr = open(model)
        return pickle.load(fr)

    def checkOneByOne(self,precision=5,X_test = None,y_test = None,model = None):
        erroNum = 0

        y_predict = model.predict(X_test)
        instance = ''
        for i in range(0,len(y_predict)):
            # instance+=str(y_test[i])+'  '+str(y_predict[i])+'\n'
            print  y_test[i],y_predict[i]
            if abs(y_test[i]-y_predict[i])>y_test[i]/100*precision:
                erroNum+=1
        
        open('predict_price.csv','wb').write(instance)
        print 
        print 'errorNum',erroNum,
        print 'allNum',len(y_predict)
        print 'precision:',(1-1.*erroNum/len(y_predict))*100
        allNum=len(y_predict)
        precision=(1-1.*erroNum/len(y_predict))*100
        # return erroNum,allNum,precision

        # df=pd.read_csv("C:/danhua/add_score/predict_price/predict_price_series_1_11_1.csv")
        # df["y_predict"]=y_predict
        # df.to_csv("C:/danhua/add_score/predict_price/predict_price_series_1_11_1.csv",index=False)

if __name__=='__main__':
    
    sys.stdout = open('C:/danhua/add_score/log/my4.log', 'a+')
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                      datefmt='%a, %d %b %Y %H:%M:%S',
                      filename='my4.log',
                      filemode='a+',stream=sys.stdout)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)                         
    

    logging.info('This is info message')

    ln = aoTest()
    X,y = ln.load_data("C:/danhua/add_score/fea_test/fea_da_car_series_2_11.csv")
    model = ln.load_model('C:/danhua/add_score/model/da_car_model_2_11.model')                        
    ln.checkOneByOne(precision=10,X_test=X,y_test =y,model=model)



     
            # y_predict = model.predict(X_test)
            # print j
            # print 'errorNum', erronum,
            # print 'allNum', allNum
            # print 'precision:', precision
            # num.append(j)
            # errorNum.append(erroNum)
            # allNum.append(allNum)
            # precision.append(precision)

            
    # df = pd.DataFrame([num, errorNum, allNum, precision]).T
    # df = df.rename(columns={0: "series_id", 1: "errorNum",
    #                         2: "allNum", 3: "precision"})   
    # print df
    # df.to_csv("C:/danhua/analysise/testnumber/allResult",index=False)
    # sys.stdout = open('C:/danhua/analysise/log/my2.log', 'a+')
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    #                     datefmt='%a, %d %b %Y %H:%M:%S',
    #                     filename='my2.log',
    #                     filemode='a+',
    #                     stream=sys.stdout)


    # logging.info('This is info message')

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)

 
