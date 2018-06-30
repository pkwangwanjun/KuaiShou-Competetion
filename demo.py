# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import multiprocessing
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
##import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

import sys
sys.path.append('/Users/wanjun/Desktop/LightGBM/python-package')
import lightgbm as lgb

import mlxtend


#使用day-15:day作为训练
#day+1:day+7作为标签
def extract_feature(day=30,training=True):

    #获取数据
    user_reg_raw=pd.read_csv('./data/user_register_log.txt',sep='\t',index_col=None,header=None)
    user_act_raw=pd.read_csv('./data/user_activity_log.txt',sep='\t',index_col=None,header=None)
    user_app_launch_raw=pd.read_csv('./data/app_launch_log.txt',sep='\t',index_col=None,header=None)
    user_video_raw=pd.read_csv('./data/video_create_log.txt',sep='\t',index_col=None,header=None)


    #选择抽取特征的天数
    user_reg=user_reg_raw[(user_reg_raw[1]<=day)]
    user_act=user_act_raw[(user_act_raw[1]<=day) & (user_act_raw[1]>=day-19)]
    user_app_launch=user_app_launch_raw[(user_app_launch_raw[1]<=day) & (user_app_launch_raw[1]>=day-19)]
    user_video=user_video_raw[(user_video_raw[1]<=day) & (user_video_raw[1]>=day-19)]


    feature_label=pd.DataFrame(index=user_reg[0])

    #改变index
    user_reg.set_index(0,inplace=True)
    user_act.set_index(0,inplace=True)
    user_app_launch.set_index(0,inplace=True)
    user_video.set_index(0,inplace=True)

    #col命名
    user_reg.columns=['reg_day','reg_type','device_type']
    user_act.columns=['act_day','page','video_id','author_id','action_type']
    user_app_launch.columns=['lanuch_day']
    user_video.columns=['create_video_day']

    #抽取launch的特征

    user_launch_day=pd.DataFrame(user_app_launch.groupby(level=0).apply(lambda x:np.unique(x)),columns=['launch_day'])

    user_launch_day['last_launch_day']=user_launch_day['launch_day'].apply(lambda x:x[-1])

    temp1=user_reg.join(user_launch_day,how='right')[['reg_day','launch_day','last_launch_day']]

    feature_label['launch_day_num']=temp1['launch_day'].apply(lambda x:len(x))
    feature_label['launch_day_ratio']=temp1['launch_day'].apply(lambda x:len(x))/(1/(1+math.e**(-(day-temp1['reg_day']+1))))
    feature_label['launch_day_last']=(day-temp1['last_launch_day']+1)/(day-temp1['reg_day']+1)

    feature_label['launch_decay']=temp1['launch_day'].apply(lambda x:sum(1./(day-x+1)))

    user_app_launch.sort_values('lanuch_day',inplace=True)

    feature_label['continuous_day']=user_app_launch.reset_index().groupby(by=0).apply(lambda x:x['lanuch_day'].diff()).groupby(level=0).apply(lambda x:(x==1).sum())

    feature_label['continuous_day/reg_day']=feature_label['continuous_day']/(day-temp1['reg_day']+1)


    #抽取video的特征

    temp2=user_video.groupby(level=0).apply(lambda x:x.groupby(by='create_video_day').apply(lambda x:len(x)))
    feature_label['video_mean']=temp2.groupby(level=0).apply(lambda x:x.mean())
    feature_label['video_std']=temp2.groupby(level=0).apply(lambda x:x.std())
    feature_label['video_sum']=temp2.groupby(level=0).apply(lambda x:x.sum())

    temp3=pd.DataFrame(user_video.groupby(level=0).apply(lambda x:np.unique(x)),columns=['video_day'])
    temp3['last_video_day']=temp3['video_day'].apply(lambda x:x[-1])
    temp3=user_reg.join(temp3,how='right')[['reg_day','video_day','last_video_day']]

    feature_label['video_day_sum']=temp3['video_day'].apply(lambda x:len(x))
    feature_label['video_day_ratio']=temp3['video_day'].apply(lambda x:len(x))/(1/(1+math.e**(-(day-temp3['reg_day']+1))))
    feature_label['video_day_last']=(day-temp3['last_video_day']+1)/(day-temp3['reg_day']+1)

    #抽取act的特征

    temp4=pd.DataFrame(user_act.groupby(level=0).apply(lambda x:x.groupby(by='act_day').apply(lambda x:len(x))),columns=['act_num'])
    temp4.reset_index(inplace=True)
    temp4.set_index(0,inplace=True)

    feature_label['decay_sum']=pd.DataFrame(temp4.groupby(level=0).apply(lambda x:sum(x['act_num']/(day-x['act_day']+1))),columns=['decay_sum'])
    feature_label['act_mean']=temp4['act_num'].groupby(level=0).apply(lambda x:x.mean())
    feature_label['act_std']=temp4['act_num'].groupby(level=0).apply(lambda x:x.std())
    feature_label['act_mean1']=temp4['act_num'].groupby(level=0).apply(lambda x:x.sum())/(1/(1+math.e**(-temp4['act_day'].groupby(level=0).apply(lambda x:len(x)))))

    #To do 算比例
    feature_label['act_type_1']=user_act['action_type'].groupby(level=0).apply(lambda x:(x==1).sum())
    feature_label['act_type_2']=user_act['action_type'].groupby(level=0).apply(lambda x:(x==2).sum())
    feature_label['act_type_3']=user_act['action_type'].groupby(level=0).apply(lambda x:(x==3).sum())
    feature_label['act_type_4']=user_act['action_type'].groupby(level=0).apply(lambda x:(x==4).sum())
    feature_label['act_type_5']=user_act['action_type'].groupby(level=0).apply(lambda x:(x==5).sum())

    #feature_label['page_0']=user_act['page'].groupby(level=0).apply(lambda x:(x==0).sum())
    #feature_label['page_1']=user_act['page'].groupby(level=0).apply(lambda x:(x==1).sum())
    #feature_label['page_2']=user_act['page'].groupby(level=0).apply(lambda x:(x==2).sum())
    #feature_label['page_3']=user_act['page'].groupby(level=0).apply(lambda x:(x==3).sum())
    #feature_label['page_4']=user_act['page'].groupby(level=0).apply(lambda x:(x==4).sum())


    #抽取reg特征

    feature_label['reg_type']=user_reg['reg_type']

#获取label

    if training:
        user_reg=pd.DataFrame(index=user_reg_raw[(user_reg_raw[1]<=day)][0])
        user_reg['label']=0
        user_act=user_act_raw[(user_act_raw[1]<=day+7) & (user_act_raw[1]>day)]
        user_app_launch=user_app_launch_raw[(user_app_launch_raw[1]<=day+7) & (user_app_launch_raw[1]>day)]
        user_video=user_video_raw[(user_video_raw[1]<=day+7) & (user_video_raw[1]>day)]

        user_reg.loc[np.intersect1d(np.array(user_reg.index),np.unique(user_act[0])),'label']=1
        user_reg.loc[np.intersect1d(np.array(user_reg.index),np.unique(user_app_launch[0])),'label']=1
        user_reg.loc[np.intersect1d(np.array(user_reg.index),np.unique(user_video[0])),'label']=1

        label=feature_label.join(user_reg,how='outer')['label']
        feature=feature_label.join(user_reg,how='outer')[feature_label.columns[feature_label.columns!='label']]
        return feature,label
    else:
        user_reg=pd.DataFrame(index=user_reg_raw[(user_reg_raw[1]<=day)][0])
        user_reg['label']=0

        label=feature_label.join(user_reg,how='outer')['label']
        feature=feature_label.join(user_reg,how='outer')[feature_label.columns[feature_label.columns!='label']]
        return feature
    #最大登陆天数连续登陆天数 TO DO


def feature_label_old(day=28,training=True):
    df,userid=extract_feature(day)

    user_act=pd.read_csv('./data/user_activity_log.txt',sep='\t',index_col=None,header=None)
    if training:
        user_act=user_act[(user_act[1]>day) & (user_act[1]<=(day+7))]
        userid_act=pd.DataFrame(user_act[0].drop_duplicates())

        userid_act[1]=1
        userid_act.set_index(0,inplace=True)
        userid_act.columns=['label']
        #df.merge(userid_act,left_index=True,how='left',right_on=0)
        feature_label=df.join(userid_act,how='left')

    else:
        userid_act=pd.DataFrame(user_act[0].drop_duplicates())

        userid_act[1]=1
        userid_act.set_index(0,inplace=True)
        userid_act.columns=['label']
        #df.merge(userid_act,left_index=True,how='left',right_on=0)
        feature_label=df.join(userid_act,how='left')

    feature_label['mean']=feature_label['act_num']/(1/(1+math.e**(-feature_label['act_days'])))
    feature_label['last']=(day-feature_label['last_day']+1)/(day-feature_label['reg_day']+1)
    feature_label['ratio']=feature_label['act_days']/(1+math.e**(-(day-feature_label['reg_day']+1)))
    #feature_label['ratio']=feature_label['act_days']/(day-feature_label['reg_day']+1)
    feature_label['mean_video']=feature_label['video_num']/(1/(1+math.e**(-feature_label['video_days'])))
    feature_label['video_ratio']=feature_label['video_days']/(day-feature_label['reg_day']+1)

    #temp=feature_label[['label','mean','last','ratio','video_num','decay_sum','mean_video','video_ratio']]
    temp=feature_label[['label','mean','last','ratio','mean_video','decay_sum','continuous_day']]

    temp.fillna(0,inplace=True)

    if training:
        return temp[temp.columns[1::]],temp.label
    else:
        return temp[temp.columns[1::]]


'''
x=feature_label[['mean','last','ratio','mean_video','decay_sum']].values
y=feature_label['label'].values
'''
'''
user_act=pd.read_csv('./data/user_activity_log.txt',sep='\t',index_col=None,header=None)
user_act.set_index(0,inplace=True)
user_act.sort_index(inplace=True)

user_act.groupby(level=0).apply(lambda x:x.groupby(by=1).apply(lambda x:len(x)))
'''


def rfmodel(x,y):
    #rf=RandomForestClassifier(n_estimators=500,verbose=1,oob_score=False,max_depth=8,class_weight={0:1,1:1.41},random_state=0)
    rf=RandomForestClassifier(n_estimators=480,verbose=1,oob_score=False,max_depth=10,class_weight={0:1,1:1.41},random_state=0,min_samples_split=100)
    scores=cross_validate(estimator=rf,X=x,y=y,cv=10,scoring=make_scorer(f1_score),n_jobs=-1)
    print(scores['test_score'].mean())
    rf.fit(x,y)
    return scores,rf
    #print(rf.oob_score_)


def gbmodel(x,y):
    gbdt=GradientBoostingClassifier(learning_rate=0.01,n_estimators=500,subsample=0.9,random_state=0)
    scores=cross_validate(estimator=gbdt,X=x,y=y,cv=10,scoring=make_scorer(f1_score),n_jobs=-1)
    print(scores['test_score'].mean())

'''
np.stack([x[:,:-1],x[:,-1].astype(np.int).reshape(-1,1)],axis=1)
x[:,:-1]
x[:,-1].astype(np.int)
#转换为类别类型
x_train.reg_type=x_train.reg_type.astype('category')
'''


def lgbmodel(x,y):
    #train_data = lgb.Dataset(data, label=label)
    lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,num_leaves=70,class_weight={0:1,1:1.41},learning_rate=0.01,subsample=0.9,sub_feature=0.9,random_state=0,n_jobs=1,objective='binary')
    scores=cross_validate(estimator=lgbc,X=x,y=y,cv=10,scoring=make_scorer(f1_score),n_jobs=4,verbose=-1)
    print(scores['test_score'].mean())
    print(scores['train_score'].mean()-scores['test_score'].mean())
    return scores,lgbc


def xgbmodel(data,label):
    clf_xgb = xgb.XGBClassifier(base_score=0.45, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.01, max_delta_step=0, max_depth=5,
           min_child_weight=1, missing=None, n_estimators=500, nthread=-1,
           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
           scale_pos_weight=1, random_state=0, silent=True, subsample=0.9)
    scores=cross_validate(estimator=clf_xgb,X=data,y=label,cv=10,scoring=make_scorer(f1_score))
    print(scores['test_score'].mean())
    return scores

def badcase(x,y):
    #rf=RandomForestClassifier(n_estimators=500,verbose=1,oob_score=False,max_depth=8,class_weight={0:1,1:1.41},random_state=0)
    gbdt=GradientBoostingClassifier(learning_rate=0.01,n_estimators=600,subsample=0.8,random_state=0)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
    #rf.fit(x_train,y_train)
    gbdt.fit(x_train,y_train)

    #pred_prob=rf.predict_proba(x_test)
    pred_prob=gbdt.predict_proba(x_test)
    #pred=rf.predict(x_test)
    pred=np.ones(len(pred_prob))
    pred[pred_prob[:,0]>0.61]=0
    print(recall_score(y_test,pred))
    print(precision_score(y_test,pred))
    print(f1_score(y_test,pred))
    badpred=y_test.loc[(pred!=y_test)]
    badpred.to_csv('bad.csv')

def baseline():
    user_act=pd.read_csv('./data/user_activity_log.txt',sep='\t',index_col=None,header=None)
    #最简单baseline
    temp=user_act[user_act[1]>=24]
    temp.reset_index(inplace=True)
    id=temp.drop_duplicates(0)[0]
    statis=temp.groupby(by=0).apply(lambda x:len(x))

    statis=pd.DataFrame(statis)
    statis.colimns=['act_num']
    statis.reset_index(inplace=True)
    statis.columns=['id','act_num']

    statis[statis['act_num']<10]['id']

    user_act.set_index(0,inplace=True)
    out=user_act.loc[statis[statis['act_num']<10]['id']]

    out_1=out[1].groupby(level=0).apply(lambda x:len(np.unique(x)))


    df=pd.DataFrame(list(set(statis['id'])-set(out_1[out_1<2].index)))

def select_custom(df):
    #一级死忠粉：reg_day+act_days==31,且reg_day<=26
    df['reg+act']=df['reg_day']+df['act_days']
    temp1=df[(df['reg+act']==31) & (df['reg_day']<=26)]



def subpred(x,y):
    x.reg_type=x.reg_type.astype('category')
    y=y.label
    clfs=[]
    for i in range(16):
        clfs.append(clone(lgb.LGBMClassifier(n_estimators=520,max_depth=-1,num_leaves=70,class_weight={0:1,1:1.4},learning_rate=0.01,subsample=0.9,sub_feature=0.9,random_state=i*111,n_jobs=1,objective='binary')))

    eclf = EnsembleVoteClassifier(clfs=clfs,voting='soft')
    eclf.fit(x,y)

    x_test=extract_feature(day=30,training=False)
    x_test.reg_type=x_test.reg_type.astype('category')
    pred=lgbc.predict(x_test)
    pred=pd.DataFrame(x_test.iloc[pred==1].index)
    #pred=pd.DataFrame(pred,index=x_test.index)
    pred.to_csv('predict.csv',columns=None,header=None)


def gridcv(x,y):
    param={'n_estimators':[480,500,520,540],'num_leaves':[65,70,85],'class_weight':[{0:1,1:1.40},{0:1,1:1.41},{0:1,1:1.415},{0:1,1:1.40},{0:1,1:1.42},{0:1,1:1.395}],'min_child_samples':[20,40,60,80,100]}
    lgbc=lgb.LGBMClassifier(max_depth=-1,num_leaves=65,learning_rate=0.01,subsample=0.8,sub_feature=0.8,random_state=0,n_jobs=-1,objective='binary',categorical_feature=26)
    #rf=RandomForestClassifier(verbose=1,oob_score=False,random_state=0)
    gs=GridSearchCV(estimator=lgbc,n_jobs=-1,verbose=1,param_grid=param,scoring=make_scorer(f1_score),cv=10)
    gs.fit(x,y)
    return gs

def stack_feature():
    temp_x=pd.DataFrame()
    temp_y=pd.DataFrame()
    for i in [20,23]:
        print('index:{}'.format(i))
        x,y=extract_feature(day=i)
        temp_x=temp_x.append(x)
        temp_y=temp_y.append(pd.DataFrame(y))

    #scores,clf=lgbmodel(temp_x,temp_y)
    return temp_x,temp_y

def vote_ensemble(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1)
    clf_lst=[]
    pred_lst=[]
    pred_sum=np.ones((len(y_test),2))
    for i in range(16):
        lgbc=lgb.LGBMClassifier(n_estimators=500,max_depth=-1,num_leaves=70,class_weight={0:1,1:1.41},learning_rate=0.01,subsample=0.9,sub_feature=0.9,random_state=i,n_jobs=4,objective='binary')
        lgbc.fit(x_train,y_train)
        pred_sum+=lgbc.predict_proba(x_test)
        print(f1_score(y_test,lgbc.predict(x_test)))
        clf_lst.append(clone(lgbc))
        pred_lst.append(f1_score(y_test,lgbc.predict(x_test)))


    pred=np.argmax(pred_sum,axis=1)
    print(f1_score(y_test,pred))
    return clf_lst,pred_lst



if __name__=='__main__':
    #x_train,y_train=extract_feature(day=23)
    x_train,y_train=stack_feature()
    x_train.reg_type=x_train.reg_type.astype('category')
    #lgbmodel(x_train,y_train)
    #x_test=extract_feature(day=30,training=False)
    '''
    rf=RandomForestClassifier(n_estimators=480,verbose=1,oob_score=False,max_depth=10,class_weight={0:1,1:1.41},random_state=0,min_samples_split=100)
    scores=cross_validate(estimator=rf,X=x,y=y,cv=10,scoring=make_scorer(f1_score),n_jobs=-1)
    print(scores['test_score'].mean())
    x3=feature_label(day=30,training=False)
    '''
    #gs=gridcv(x,y)
    #scores,rf=rfmodel(x,y)
    #gbmodel(x,y)
    '''
    rf=rfmodel(x,y)
    x=feature_label(day=30,training=False)
    rf.predict(x)
    '''
