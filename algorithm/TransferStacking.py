import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
# Transfer Stacking
def Ffold_cross_val(Xtrain, Ytrain, F, estimator):
# KFold
    row_Xtrain = Xtrain.shape[0]
    # 创建一个 KFold 对象，设置折数为 5
    kf = KFold(n_splits=F)
    # 使用 KFold 对象划分数据集，并进行交叉验证
    output = np.zeros(row_Xtrain, 1)
    for train_index, val_index in kf.split(Xtrain):
        x_train, x_val = [Xtrain[i] for i in train_index], [Xtrain[i] for i in val_index]
        y_train, _ = [Ytrain[i] for i in train_index], [Ytrain[i] for i in val_index]
        estimator.fit(x_train, y_train)
        output[val_index] = estimator.predict(x_val)
    return output

def TransferStacking(Xmsource, Xtarget, Ymsource, Ytarget, Xtest, *kargs):
    """
    Transfer Stacking
    parameter
    _________
    Xmsource = dict{source1: A,
                    source2: B,
                    ...}
    Xtarget - matrix
    Ymsource = dict{source1: A,
                    source2: B.
                    ...}
    Ytarget - vector
    Xtest

    Atrributes
    ----------
    
    return
    ------
    """
    Xmsource = list(Xmsource.values())
    Xsource = np.concatenate(Xmsource)
    Xtrain = np.concatenate([Xsource, Xtarget], axis=0)
    Ymsource = list(Ymsource.value())
    Ysource = np.concatenate(Ymsource)
    Ytrain = np.concatenate([Ysource, Ytarget], axis=0)
    
    num_source = len(Xmsource)
    row_Xtrain = Xtrain.shape[0]
    output = np.zeros(row_Xtrain, num_source)
    estimators = []
    for i in range(num_source):
        estimator = DecisionTreeClassifier(criterion='gini',max_depth=3, random_state=42)
        estimator.fit(Xmsource[i], Ymsource[i])
        output[:,i] = estimator.predict(Xtrain)
        estimators.append(estimator)
    
    reg = DecisionTreeClassifier(max_depth=2,splitter='random',max_features="log2",random_state=0)
    estimators.append(reg)

    output_cv = Ffold_cross_val(Xtrain, Ytrain, 5, reg)
    meta_feature = np.concatenate([output, output_cv], axis=1)
    

    linearR = LinearRegression()
    linearR.fit(meta_feature, Ytrain)
    print('The linear combination of hypothesis is founded:')
    print('coef:', linearR.coef_ ,'|| intercept :', linearR.intercept_)
    hypothesis = np.zeros(row_Xtrain, len(estimators))
    for j in range(len(estimators)):
        hypothesis[:,j] = estimators[j].predict(Xtest)

    coef = linearR.coef_
    intercept = linearR.intercept_
    predict = np.ones(row_Xtrain)*intercept
    for i in range(len(coef)):
        predict += coef[j]*hypothesis[:,j]
    return predict