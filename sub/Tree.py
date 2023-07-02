import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import ensemble
import graphviz 
import matplotlib.pyplot as plt


class decision:
    def __init__(self,X): #initialization
        self.X=X #matrix A
        self.Nr=X.shape[0] #number of rows
        self.Nc=X.shape[1] #number of columns
        return
    def shuffle(self):
        X=self.X
        Nr=self.Nr
        Nc=self.Nc
        indexsh=np.arange(Nr) #make a new matrix with size of Np
        np.random.shuffle(indexsh)# shuffle the matrix indexsh
        Xsh=X.copy(deep=True)
        Xsh=Xsh.set_axis(indexsh,axis=0,inplace=False)
        Xsh=Xsh.sort_index(axis=0)
        Ntr=158 # number of training points
        Nte=Nr-Ntr   # number of test points
        X_tr=Xsh[0:Ntr]
        X_te=Xsh[Ntr:]
        return X_tr,X_te
    def plot_tree(self,feat_names,X_train,X_test):
        X_tr=X_train
        X_te=X_test
        target_names = ['notckd','ckd']
        labels = X_tr.loc[:,'classk']
        data = X_tr.drop('classk', axis=1)
        clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)
        clfXtrain = clfXtrain.fit(data,labels)
        test_pred = clfXtrain.predict(X_te.drop('classk', axis=1))
        from sklearn.metrics import accuracy_score
        print('accuracy =', accuracy_score(X_te.loc[:,'classk'],test_pred))
        from sklearn.metrics import confusion_matrix
        print('Confusion matrix')
        conf_matrix=confusion_matrix(X_te.loc[:,'classk'],test_pred)
        print(conf_matrix)
        True_positive=conf_matrix[0,0]
        Totall_positive=conf_matrix[0,0]+conf_matrix[0,1]
        sens=True_positive/Totall_positive
        print("sensitivity:")
        print(round(sens,3))
        True_negative=conf_matrix[1,1]
        Totall_negative=conf_matrix[1,0]+conf_matrix[1,1]
        spec=True_negative/Totall_negative
        print("Specificity:")
        print(round(spec,3))
        # tree.plot_tree(clfXtrain)
        # plt.show() 
        # #text option
        # text_representation = tree.export_text(clfXtrain)
        # print(text_representation)
        #option with colors
        fig = plt.figure(figsize=(4,4))
        tree.plot_tree(clfXtrain,
                            feature_names=feat_names[:24],
                            class_names=target_names,
                            filled=True, rounded=True)
        plt.show() 
        return
    def forrest(self,feat_names,X_train,X_test):
        X_tr=X_train
        X_te=X_test
        target_names = ['notckd','ckd']
        labels = X_tr.loc[:,'classk']
        data = X_tr.drop('classk', axis=1)
        clfXtrain = ensemble.RandomForestClassifier(criterion='entropy',random_state=4)
        clfXtrain = clfXtrain.fit(data,labels)
        test_pred = clfXtrain.predict(X_te.drop('classk', axis=1))
        from sklearn.metrics import accuracy_score
        print('accuracy =', accuracy_score(X_te.loc[:,'classk'],test_pred))
        from sklearn.metrics import confusion_matrix
        print('Confusion matrix')
        conf_matrix=confusion_matrix(X_te.loc[:,'classk'],test_pred)
        print(conf_matrix)
        True_positive=conf_matrix[0,0]
        Totall_positive=conf_matrix[0,0]+conf_matrix[0,1]
        sens=True_positive/Totall_positive
        print("sensitivity:")
        print(round(sens,3))
        True_negative=conf_matrix[1,1]
        Totall_negative=conf_matrix[1,0]+conf_matrix[1,1]
        spec=True_negative/Totall_negative
        print("Specificity:")
        print(round(spec,3))
        return


