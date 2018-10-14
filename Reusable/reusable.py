# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:31:06 2018

@author: Bhawani Singh
"""

'''feature worth OHE'''
def ohe(X_train,X_val, ohe_feat):
    OHE = OneHotEncoder(sparse=False)
    if len(ohe_feat) ==1:
        OHE.fit(pd.concat([X_train[ohe_feat], X_val[ohe_feat]], axis=0).values.reshape(-1,1))
        return OHE.transform(X_train[ohe_feat].values.reshape(-1,1)), OHE.transform(X_val[ohe_feat].values.reshape(-1,1))
    else:
        OHE.fit(pd.concat([X_train[ohe_feat], X_val[ohe_feat]], axis=0))
        return OHE.transform(X_train[ohe_feat]), OHE.transform(X_val[ohe_feat])
    
    
def fw_feature_selection(X_train,y_train, X_val, y_val, loss=True): #need to drop last feat
    score_hist = []
    selected_feature = set([])
    while len(score_hist) < 2 or (score_hist[-1][0] < score_hist[-2][0]) == loss:  #loss
        score = []
        
        for f in X_train.columns.tolist():  #if passing ndarray then convert to feat index
            if f not in selected_feature:        
                ohe_feat = list(selected_feature) + [f]
                
                X_ohe_train, X_ohe_val = ohe(X_train,X_val, ohe_feat)
                
                lr = lm.LinearRegression()
                lr.fit(X_ohe_train, y_train)
                val_score = np.sqrt(mean_squared_error(y_val,lr.predict(X_ohe_val)))
                
                score.append([val_score, f])
        
        score_hist.append(sorted(score, reverse=loss)[-1])
        selected_feature.add(sorted(score, reverse=loss)[-1][1])
        
    #droping last feature
    selected_feature.remove((score_hist[-1][1]))
    return selected_feature, score_hist