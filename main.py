import pandas as pd
import numpy as np
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt
import sub.Tree as Tree

# define the feature names:
feat_names=['age','bp','sg','al','su','rbc','pc',
'pcc','ba','bgr','bu','sc','sod','pot','hemo',
'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
'ane','classk']
ff=np.array(feat_names)
feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
         'num','num','num','num','num','num','num','num','num',
         'cat','cat','cat','cat','cat','cat','cat'])
xx=pd.read_csv("./data/chronic_kidney_disease_v2.arff",sep=',',
    skiprows=29,names=feat_names, 
    header=None,na_values=['?','\t?'],)
Np,Nf=xx.shape
#%% change categorical data into numbers:
key_list=["normal","abnormal","present","notpresent","yes",
"no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
xx=xx.replace(key_list,key_val)
print(xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values

#%% manage the missing data through regression
print(xx.info())
x=xx.copy()
# drop rows with less than 19=Nf-6 recorded features:
x=x.dropna(thresh=19)
x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
n=x.isnull().sum(axis=1)# check the number of missing values in each row
print('max number of missing values in the reduced dataset: ',n.max())
print('number of points in the reduced dataset: ',len(n))
# take the rows with exctly Nf=25 useful features; this is going to be the training dataset
# for regression
Xtrain=x.dropna(thresh=25)
Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
# get the possible values (i.e. alphabet) for the categorical features
alphabets=[]
for k in range(len(feat_cat)):
    if feat_cat[k]=='cat':
        val=Xtrain.iloc[:,k]
        val=val.unique()
        alphabets.append(val)
    else:
        alphabets.append('num')

#%% run regression tree on all the missing data
#normalize the training dataset
mm=Xtrain.mean(axis=0)
ss=Xtrain.std(axis=0)
Xtrain_norm=(Xtrain-mm)/ss
# get the data subset that contains missing values 
Xtest=x.drop(x[x.isnull().sum(axis=1)==0].index)
Xtest.reset_index(drop=True, inplace=True)# reset the index of the dataframe
Xtest_norm=(Xtest-mm)/ss # nomralization
Np,Nf=Xtest_norm.shape
regr=tree.DecisionTreeRegressor() # instantiate the regressor
for kk in range(Np):
    xrow=Xtest_norm.iloc[kk]#k-th row
    mask=xrow.isna()# columns with nan in row kk
    Data_tr_norm=Xtrain_norm.loc[:,~mask]# remove the columns from the training dataset
    y_tr_norm=Xtrain_norm.loc[:,mask]# columns to be regressed
    regr=regr.fit(Data_tr_norm,y_tr_norm)
    Data_te_norm=Xtest_norm.loc[kk,~mask].values.reshape(1,-1) # row vector
    ytest_norm=regr.predict(Data_te_norm)
    Xtest_norm.iloc[kk][mask]=ytest_norm # substitute nan with regressed values
Xtest_new=Xtest_norm*ss+mm # denormalize
# substitute regressed numerical values with the closest element in the alphabet
index=np.argwhere(feat_cat=='cat').flatten()
for k in index:
    val=alphabets[k].flatten() # possible values for the feature
    c=Xtest_new.iloc[:,k].values # values in the column
    c=c.reshape(-1,1)# column vector
    val=val.reshape(1,-1) # row vector
    d=(val-c)**2 # matrix with all the distances w.r.t. the alphabet values
    ii=d.argmin(axis=1) # find the index of the closest alphabet value
    Xtest_new.iloc[:,k]=val[0,ii]
print(Xtest_new.nunique())
print(Xtest_new.describe().T)

#Shuffle the data
X_new= pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)
a=Tree.decision(X_new)
a.plot_tree(feat_names,Xtrain,Xtest_new)

#Decision Tree 1
np.random.seed(301000) # set the seed for random shuffling (student number)
b=Tree.decision(X_new)
X_tr,X_te=b.shuffle()
print("____________________________________________")
print("Decision Tree 1_marticolar number")
b.plot_tree(feat_names,X_tr,X_te)
print("random Forest")
b.forrest(feat_names,X_tr,X_te)

#Decision Tree 2
np.random.seed(301000) # set the seed for random shuffling
c=Tree.decision(X_new)
X_tr,X_te=c.shuffle()
print("____________________________________________")
print("Decision Tree 2")
c.plot_tree(feat_names,X_tr,X_te)         

#Decision Tree 3
np.random.seed(301000) # set the seed for random shuffling
d=Tree.decision(X_new)
X_tr,X_te=d.shuffle()
print("____________________________________________")
print("Decision Tree 3")
d.plot_tree(feat_names,X_tr,X_te)             