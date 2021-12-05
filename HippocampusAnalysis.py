#import
import pandas as pd
import numpy as np
from numpy import arange
from pandas import read_csv
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn import metrics
import regressors
from regressors import stats
from numpy import arange
from pandas import read_csv

#split data into features and predictor
def splitdata(df, feat, predictors):   
    y = df[feat]
    X = df[predictors]
    return X, y

def splitdata_normalize(df, feat, predictors):  
    df2= df[[feat] + predictors] #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    xy_scaled = min_max_scaler.fit_transform(df2.values)
    df_scaled = pd.DataFrame(xy_scaled, columns=df2.columns)
    y = df_scaled[feat]
    X = df_scaled[predictors]
    #X_int = pd.DataFrame(np.concatenate( ( np.ones((X.shape[0], 1)), X), axis = 1 ), columns = ['intercept'] + predictors)
    return X, y

def normalize_df(df):  
    min_max_scaler = preprocessing.MinMaxScaler()
    xy_scaled = min_max_scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(xy_scaled, columns=df.columns)
    return df_scaled

def getcoeffandpvals(model, X, y):
    coefs = [model.intercept_] + list(model.coef_)
    pvals = pd.DataFrame(stats.coef_pval(model, X, y),  (['intercept']+X.columns.to_list()), columns=['pval'])
    coeff_df = pd.DataFrame(coefs, (['intercept']+X.columns.to_list()), columns=['coeff'])
    return pd.concat([coeff_df, pvals], axis=1)

def runLinReg(df, feat, predictors):
    X, y = splitdata_normalize(df, feat, predictors)

    # Code from try1
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)

    linreg = LinearRegression(fit_intercept=True)
    linreg.fit(X_train, y_train)
    train_results = getcoeffandpvals(linreg, X_train, y_train)
    
    y_pred = linreg.predict(X_test)
    actualvspred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
    print(actualvspred.sort_values(by='Difference', ascending=False))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return linreg, X_train, X_test, y_train, y_test, actualvspred, train_results


def runRidgeCVReg(df, feat, predictors, alpha=1):
    X, y = splitdata_normalize(df, feat, predictors)

    # Code from try1
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)

    linreg = RidgeCV(fit_intercept=True, alpha=alpha)
    linreg.fit(X_train, y_train)
    train_results = getcoeffandpvals(linreg, X_train, y_train)
    
    y_pred = linreg.predict(X_test)
    actualvspred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
    print(actualvspred.sort_values(by='Difference', ascending=False))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return linreg, X_train, X_test, y_train, y_test, actualvspred, train_results



#adapted from https://machinelearningmastery.com/ridge-regression-with-python/
def optimizeRidge(df, feat, predictors):
    # define model
    X, y = splitdata_normalize(df, feat, predictors)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
    # fit model
    model.fit(X, y)
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)


def runLinRegLasso(df, feat, predictors, alpha=1):
    X, y = splitdata_normalize(df, feat, predictors)

    # Code from try1
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)

    linreg = Lasso(fit_intercept=True, alpha=alpha)
    linreg.fit(X_train, y_train)
    train_results = getcoeffandpvals(linreg, X_train, y_train)
    
    y_pred = linreg.predict(X_test)
    actualvspred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
    print(actualvspred.sort_values(by='Difference', ascending=False))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return linreg, X_train, X_test, y_train, y_test, actualvspred, train_results


#adapted from https://machinelearningmastery.com/lasso-regression-with-python/
def optimizeLasso(df, feat, predictors):
    # define model
    X, y = splitdata_normalize(df, feat, predictors)

    model = Lasso()
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = arange(0, 1, 0.01)
    # define search
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X, y)
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)


#import data
df_all = pd.read_excel('CondensedDataandKey.xlsx',sheet_name='data')

#all cognitive variables

AllCogAge = ['PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'ListSort_AgeAdj']
AllCogUnadj = ['PicSeq_Unadj', 'CardSort_Unadj',	'Flanker_Unadj', 'ReadEng_Unadj',	'PicVocab_Unadj', 'ProcSpeed_Unadj', 'ListSort_Unadj']

#all sleep/brain variables
main_vars = ['Subject', 'Gender', 'Age', 'FS_IntraCranial_Vol', 'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol','PSQI_Score', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7','CogFluidComp_Unadj','CogFluidComp_AgeAdj','CogTotalComp_Unadj', 'CogTotalComp_AgeAdj', 'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj']

df = df_all[(main_vars+AllCogAge+AllCogUnadj)]
df = df.dropna()

#data cleanup
#make gender binary numbers
df['Gender'] = [0 if i == 'M' else 1 for i in df['Gender']]

#turn age categories into ordinal variable
age_dict = {'22-25': 1, '26-30': 2, '31-35': 3, '36+': 4}
df['AgeCat'] = df['Age'].replace(age_dict)

#variables we're controlling for
control_vars = ['Gender','AgeCat','FS_IntraCranial_Vol']

#Total PSQI and controls
pred_PSQI_tot_vars = ['PSQI_Score'] + control_vars

#PSQI Component subscores (7) 
pred_PSQI_comp_vars = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'] + control_vars




#optimizing Lasso predicting left/right hippocampus volume using the total PSQI or PSQI component scores- didn't work
#optimizeLasso(df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars)
#optimizeLasso(df, 'FS_R_Hippo_Vol', pred_PSQI_tot_vars)

#optimizeLasso(df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars)
#optimizeLasso(df, 'FS_R_Hippo_Vol', pred_PSQI_comp_vars)

#optimizeLasso(df, 'PSQI_Score', pred_PSQI_comp_vars)



#Ridge Regression
TotPSQI_HippoL_linreg, TotPSQI_HippoL_X_train, TotPSQI_HippoL_X_test, TotPSQI_HippoL_y_train, TotPSQI_HippoL_y_test, TotPSQI_HippoL_actualvspred, TotPSQI_HippoL_train_results= runRidgeCVReg(
    df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars)


TotPSQI_HippoR_linreg, TotPSQI_HippoR_X_train, TotPSQI_HippoR_X_test, TotPSQI_HippoR_y_train, TotPSQI_HippoR_y_test, TotPSQI_HippoR_actualvspred, TotPSQI_HippoR_train_results= runRidgeCVReg(
    df, 'FS_R_Hippo_Vol', pred_PSQI_tot_vars)


CompPSQI_HippoL_linreg, CompPSQI_HippoL_X_train, CompPSQI_HippoL_X_test, CompPSQI_HippoL_y_train, CompPSQI_HippoL_y_test, CompPSQI_HippoL_actualvspred, CompPSQI_HippoL_train_results= runRidgeCVReg(
    df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars)
print(CompPSQI_HippoL_train_results)

CompPSQI_HippoR_linreg, CompPSQI_HippoR_X_train, CompPSQI_HippoR_X_test, CompPSQI_HippoR_y_train, CompPSQI_HippoR_y_test, CompPSQI_HippoR_actualvspred, CompPSQI_HippoR_train_results= runRidgeCVReg(
    df, 'FS_R_Hippo_Vol', pred_PSQI_comp_vars)
print(CompPSQI_HippoR_train_results)





#predicting the cognitive scores using the sleep scores - didn't work
#optimizeLasso(df, 'CogFluidComp_AgeAdj', ['Gender', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'] )
#optimizeLasso(df, 'CogTotalComp_AgeAdj', ['Gender', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'])
#optimizeLasso(df, 'CogCrystalComp_AgeAdj', ['Gender', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'])


FluidCompAge_HippoL_linreg, FluidCompAge_HippoL_X_train, FluidCompAge_HippoL_X_test, FluidCompAge_HippoL_y_train, FluidCompAge_HippoL_y_test, FluidCompAge_HippoL_actualvspred, FluidCompAge_HippoL_train_results= runRidgeCVReg(
    df, 'FS_L_Hippo_Vol', ['Gender','CogFluidComp_AgeAdj','FS_IntraCranial_Vol'])
print(FluidCompAge_HippoL_train_results)

TotCompAge_HippoL_linreg, TotCompAge_HippoL_X_train, TotCompAge_HippoL_X_test, TotCompAge_HippoL_y_train, TotCompAge_HippoL_y_test, TotCompAge_HippoL_actualvspred, TotCompAge_HippoL_train_results= runRidgeCVReg(
    df, 'FS_L_Hippo_Vol', ['Gender','CogTotalComp_AgeAdj','FS_IntraCranial_Vol'])
print(TotCompAge_HippoL_train_results)

CrystalCompAge_HippoL_linreg, CrystalCompAge_HippoL_X_train, CrystalCompAge_HippoL_X_test, CrystalCompAge_HippoL_y_train, CrystalCompAge_HippoL_y_test, CrystalCompAge_HippoL_actualvspred, CrystalCompAge_HippoL_train_results= runRidgeCVReg(
    df, 'FS_L_Hippo_Vol', ['Gender','CogCrystalComp_AgeAdj','FS_IntraCranial_Vol'])
print(CrystalCompAge_HippoL_train_results)

FluidCompAge_HippoR_linreg, FluidCompAge_HippoR_X_train, FluidCompAge_HippoR_X_test, FluidCompAge_HippoR_y_train, FluidCompAge_HippoR_y_test, FluidCompAge_HippoR_actualvspred, FluidCompAge_HippoR_train_results= runRidgeCVReg(
    df, 'FS_R_Hippo_Vol', ['Gender','CogFluidComp_AgeAdj','FS_IntraCranial_Vol'])
print(FluidCompAge_HippoR_train_results)

TotCompAge_HippoR_linreg, TotCompAge_HippoR_X_train, TotCompAge_HippoR_X_test, TotCompAge_HippoR_y_train, TotCompAge_HippoR_y_test, TotCompAge_HippoR_actualvspred, TotCompAge_HippoR_train_results= runRidgeCVReg(
    df, 'FS_R_Hippo_Vol', ['Gender','CogTotalComp_AgeAdj','FS_IntraCranial_Vol'])
print(TotCompAge_HippoR_train_results)

CrystalCompAge_HippoR_linreg, CrystalCompAge_HippoR_X_train, CrystalCompAge_HippoR_X_test, CrystalCompAge_HippoR_y_train, CrystalCompAge_HippoR_y_test, CrystalCompAge_HippoR_actualvspred, CrystalCompAge_HippoR_train_results= runRidgeCVReg(
    df, 'FS_R_Hippo_Vol', ['Gender','CogCrystalComp_AgeAdj','FS_IntraCranial_Vol'])
print(CrystalCompAge_HippoR_train_results)


AllCogAge = ['PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'ListSort_AgeAdj']
AllCogUnadj = ['PicSeq_Unadj', 'CardSort_Unadj',	'Flanker_Unadj', 'ReadEng_Unadj',	'PicVocab_Unadj', 'ProcSpeed_Unadj', 'ListSort_Unadj']

AllCogAge_HippoL_linreg, AllCogAge_HippoL_X_train, AllCogAge_HippoL_X_test, AllCogAge_HippoL_y_train, AllCogAge_HippoL_y_test, AllCogAge_HippoL_actualvspred, AllCogAge_HippoL_train_results= runRidgeCVReg(
    df, 'FS_L_Hippo_Vol', (['Gender','FS_IntraCranial_Vol']+AllCogAge))
print(AllCogAge_HippoL_train_results)

AllCogAge_HippoR_linreg, AllCogAge_HippoR_X_train, AllCogAge_HippoR_X_test, AllCogAge_HippoR_y_train, AllCogAge_HippoR_y_test, AllCogAge_HippoR_actualvspred, AllCogAge_HippoR_train_results= runRidgeCVReg(
    df, 'FS_R_Hippo_Vol', (['Gender','FS_IntraCranial_Vol']+AllCogAge))
print(AllCogAge_HippoR_train_results)



#Lasso for cogntive components predicting volume - didn't work
#optimizeLasso(df, 'FS_L_Hippo_Vol', (['Gender','FS_IntraCranial_Vol']+AllCogAge))
#optimizeLasso(df, 'FS_R_Hippo_Vol', (['Gender','FS_IntraCranial_Vol']+AllCogAge))

#optimizeLasso(df, 'FS_L_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])
#optimizeLasso(df, 'FS_R_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])


Cog2Age_HippoL_linreg, Cog2Age_HippoL_X_train, Cog2Age_HippoL_X_test, Cog2Age_HippoL_y_train, Cog2Age_HippoL_y_test, Cog2Age_HippoL_actualvspred, Cog2Age_HippoL_train_results= runRidgeCVReg(
    df, 'FS_L_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])
print(Cog2Age_HippoL_train_results)



Cog2Age_HippoR_linreg, Cog2Age_HippoR_X_train, Cog2Age_HippoR_X_test, Cog2Age_HippoR_y_train, Cog2Age_HippoR_y_test, Cog2Age_HippoR_actualvspred, Cog2Age_HippoR_train_results= runRidgeCVReg(
    df, 'FS_R_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])
print(Cog2Age_HippoR_train_results)

