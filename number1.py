#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import csv




for j in range(1,2589):
    
    data="C:/danhua/da_carset/da_car_series_%s.csv"%j
    new_train_data="C:/danhua/analysise/17/da_car_series_1.csv"%j
    train="C:/danhua/analysise/number/series_num_%s.csv"%j
    # test="C:/danhua/analysise/testnumber/series_num_%s.csv"%j
    try:    
        f=open(data)

        line=f.readline().strip()
        line=f.readline().strip() 
        model_set=set()
        while line:
            dt=line.split('\t')
            model_id=dt[1]
            model_set.add(model_id)
            line=f.readline().strip() 
# print model_set

        model_id1,test_num=[],[]
        for i in model_set:
            num=0
            g=open(test_data)
            line=g.readline().strip()
            line=g.readline().strip()
            while line:
                df=line.split(',')
                if i==df[1]:
                    num+=1
                line=g.readline().strip()
            test_num.append(num)
        da=pd.read_csv(train)
        db = pd.DataFrame([test_num]).T
        print db
        da["new_train_num"]=db
        da.to_csv(test,index=False)
    except:(AttributeError,IOError,NameError)




# g=open(data)


# for i in model_set:
    
#     line=g.readline().strip()	
#     line=g.readline().strip() 

#     while line:     
#         model_i=[]
#         model_price_i=[]
#         dt=line.split('	')    	
#         model_price=dt[12]        	
#         model_price_i.append(model_price)
#         model_i.append(i)
#         line=g.readline().strip()

#     Di=pd.DataFrame([model_i,model_price_i])
#     print Di
        # Si=Di.loc[1].astype(np.float)
        # print Si.describe()
        
# print model_id_set,model_price_set
    
# # print D
# # print D.corr(method='Spearson')     
# # 
# S1=D.loc[0].astype(np.float)
# S2=D.loc[1].astype(np.float)
# # print S1.isnull
# # print S2.isnull
# # print S1.corr(S2,method='spearman')
# sta1=S1.describe()
# sta2=S2.describe()
# # ,sta2
# sta1.loc['var']=sta1.loc['std']/sta1.loc['mean']
# sta2.loc['var']=sta2.loc['std']/sta2.loc['mean']
# print sta1,sta2