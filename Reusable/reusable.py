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
	
	
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'datetime64[ns]':
            continue
        
        if col_type != object and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
	
#specially when want to build sep model for each label
#othervise ovr
# to build model in a multi-label problem
def modelfit(S1, C1, S2, model, makeA1=False, verbosity=False):   #done
    l = C1.shape[1]
    A2 = np.zeros((S2.shape[0], l))
    if makeA1:
        A1 = np.zeros((S1.shape[0], l))
    if hasattr(model, 'predict_proba'):
        for j in range(l):
            if np.any(C1[:, j] > 0):
                model.fit(S1, C1[:, j])
                A2[:, j] = model.predict_proba(S2)[:, 1]
                if makeA1:
                    A1[:, j] = model.predict_proba(S1)[:, 1]
                if verbosity:
                    print( j)
    else:
        for j in range(l):
            if np.any(C1[:, j] > 0):
                model.fit(S1, C1[:, j])
                A2[:, j] = model.predict(S2)
                if makeA1:
                    A1[:, j] = model.predict(S1)
                if verbosity:
                    print (j)
    if makeA1:
        return A2, A1
    return A2
	
	
#data is ndarray of ascii value of char | for ascii value ord(char)	
# boundary_letter is -1 
# (-1) pre (-1) data (-1) next (-1)
def context_window_transform(data, pad_size):
    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(pad_size)]
    data = pre + data + pre
    neo_data = []
    for i in np.arange(len(data) - pad_size * 2):
        row = []
        for x in data[i : i + pad_size * 2 + 1]:
            row.append([boundary_letter])
            row.append(x)
        row.append([boundary_letter])
        neo_data.append([int(x) for y in row for x in y])
    return neo_data