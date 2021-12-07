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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing

from yellowbrick.regressor import ResidualsPlot


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


def runRidgeCVReg(df, feat, predictors):
    X, y = splitdata_normalize(df, feat, predictors)

    # Code from try1
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)

    linreg = RidgeCV(fit_intercept=True)
    linreg.fit(X_train, y_train)
    train_results = getcoeffandpvals(linreg, X_train, y_train)
    
    y_pred = linreg.predict(X_test)
    actualvspred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
    print(actualvspred.sort_values(by='Difference', ascending=False))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return linreg, X_train, X_test, y_train, y_test, actualvspred, train_results


def runRidgeReg(df, feat, predictors, alpha=1):
    X, y = splitdata_normalize(df, feat, predictors)

    # Code from try1
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)

    linreg = Ridge(fit_intercept=True, alpha=alpha)
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
    
    

#adapted from https://machinelearningmastery.com/ridge-regression-with-python/
def optimizeRidgeFull(df, feat, predictors):
    # define model
    X, y = splitdata_normalize(df, feat, predictors)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    linreg = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
    # fit model
    linreg.fit(X_train, y_train)
    # summarize chosen configuration
    opt_alpha = linreg.alpha_
    train_results = getcoeffandpvals(linreg, X_train, y_train)

    y_pred = linreg.predict(X_test)
    actualvspred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
    print(actualvspred.sort_values(by='Difference', ascending=False))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('alpha: %f' % opt_alpha)
    
    return linreg, X_train, X_test, y_train, y_test, actualvspred, train_results, opt_alpha


#same as above but larger alpha range  
def optimizeRidgeFullLargerAlphaRange(df, feat, predictors):
    # define model
    X, y = splitdata_normalize(df, feat, predictors)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    std = StandardScaler()
    X_train_std = pd.DataFrame(std.fit_transform(X_train), columns = X_train.columns)
    X_test_std = pd.DataFrame(std.transform(X_test), columns = X_test.columns)

    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    alphas = [1e-15, 1e-10, 1e-8] + list(arange(0.01, 1, 0.01)) + list(arange(1, 5, 0.2)) + list(arange(5, 40, 1))
    linreg = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_absolute_error')
    # fit model
    linreg.fit(X_train_std, y_train)
    # summarize chosen configuration
    opt_alpha = linreg.alpha_
    train_results = getcoeffandpvals(linreg, X_train_std, y_train)

    y_pred = linreg.predict(X_test_std)
    actualvspred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
    print(actualvspred.sort_values(by='Difference', ascending=False))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('alpha: %f' % opt_alpha)
    
    return linreg, X_train, X_test, y_train, y_test, actualvspred, train_results, opt_alpha



    
    
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


#all sleep/brain variables
main_vars = ['Subject', 'Gender', 'Age', 'FS_IntraCranial_Vol', 'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol','PSQI_Score', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7','CogFluidComp_Unadj','CogFluidComp_AgeAdj','CogTotalComp_Unadj', 'CogTotalComp_AgeAdj', 'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj']

df = df_all[main_vars]
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
pred_PSQI_tot_vars = ['PSQI_Score', 'Gender','AgeCat','FS_IntraCranial_Vol']

#PSQI Component subscores (7) 
pred_PSQI_comp_vars = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'] + control_vars




#optimizing Lasso predicting left/right hippocampus volume using the total PSQI or PSQI component scores- didn't work
#optimizeLasso(df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars)
#optimizeLasso(df, 'FS_R_Hippo_Vol', pred_PSQI_tot_vars)
#optimizeLasso(df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars)
#optimizeLasso(df, 'FS_R_Hippo_Vol', pred_PSQI_comp_vars)
#optimizeLasso(df, 'PSQI_Score', pred_PSQI_comp_vars)




#predicting left hipopocampus volume using total PSQI (and gender/age/ICV)
#optimizeRidge(df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars)
#get alpha = 0.13
#Ridge Regression
TotPSQI_HippoL_linreg, TotPSQI_HippoL_X_train, TotPSQI_HippoL_X_test, TotPSQI_HippoL_y_train, TotPSQI_HippoL_y_test, TotPSQI_HippoL_actualvspred, TotPSQI_HippoL_train_results= runRidgeReg(
    df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars, alpha=0.13)
#print(TotPSQI_HippoL_train_results)

#predicting right hipopocampus volume using total PSQI (and gender/age/ICV)
#optimizeRidge(df, 'FS_R_Hippo_Vol', pred_PSQI_tot_vars)
#get alpha = 0.09
#Ridge Regression
TotPSQI_HippoR_linreg, TotPSQI_HippoR_X_train, TotPSQI_HippoR_X_test, TotPSQI_HippoR_y_train, TotPSQI_HippoR_y_test, TotPSQI_HippoR_actualvspred, TotPSQI_HippoR_train_results= runRidgeReg(
    df, 'FS_R_Hippo_Vol', pred_PSQI_tot_vars, alpha=0.09)
#print(TotPSQI_HippoR_train_results)


#predicting left hipopocampus volume using 7 PSQI comp scores (and gender/age/ICV)
#optimizeRidge(df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars)
#get alpha = 0.49
#Ridge Regression
CompPSQI_HippoL_linreg, CompPSQI_HippoL_X_train, CompPSQI_HippoL_X_test, CompPSQI_HippoL_y_train, CompPSQI_HippoL_y_test, CompPSQI_HippoL_actualvspred, CompPSQI_HippoL_train_results= runRidgeReg(
    df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars, alpha=0.49)
#looks like Comp1 (subjective sleep quality) and Comp4 (sleep duration) were possibly decent predictors, but not great


#predicting left hipopocampus volume using 7 PSQI comp scores (and gender/age/ICV)
#optimizeRidge(df, 'FS_R_Hippo_Vol', pred_PSQI_comp_vars)
#get alpha = 0.18
CompPSQI_HippoR_linreg, CompPSQI_HippoR_X_train, CompPSQI_HippoR_X_test, CompPSQI_HippoR_y_train, CompPSQI_HippoR_y_test, CompPSQI_HippoR_actualvspred, CompPSQI_HippoR_train_results= runRidgeReg(
    df, 'FS_R_Hippo_Vol', pred_PSQI_comp_vars, alpha=0.18)
#print(CompPSQI_HippoR_train_results)





#predicting the cognitive scores using the sleep scores - didn't work
#optimizeLasso(df, 'CogFluidComp_AgeAdj', ['Gender', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'] )
#optimizeLasso(df, 'CogTotalComp_AgeAdj', ['Gender', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'])
#optimizeLasso(df, 'CogCrystalComp_AgeAdj', ['Gender', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'])


FluidCompAge_HippoL_linreg, FluidCompAge_HippoL_X_train, FluidCompAge_HippoL_X_test, FluidCompAge_HippoL_y_train, FluidCompAge_HippoL_y_test, FluidCompAge_HippoL_actualvspred, FluidCompAge_HippoL_train_results= runRidgeReg(
    df, 'FS_L_Hippo_Vol', ['Gender','CogFluidComp_AgeAdj','FS_IntraCranial_Vol'])
print(FluidCompAge_HippoL_train_results)

TotCompAge_HippoL_linreg, TotCompAge_HippoL_X_train, TotCompAge_HippoL_X_test, TotCompAge_HippoL_y_train, TotCompAge_HippoL_y_test, TotCompAge_HippoL_actualvspred, TotCompAge_HippoL_train_results= runRidgeReg(
    df, 'FS_L_Hippo_Vol', ['Gender','CogTotalComp_AgeAdj','FS_IntraCranial_Vol'])
print(TotCompAge_HippoL_train_results)

CrystalCompAge_HippoL_linreg, CrystalCompAge_HippoL_X_train, CrystalCompAge_HippoL_X_test, CrystalCompAge_HippoL_y_train, CrystalCompAge_HippoL_y_test, CrystalCompAge_HippoL_actualvspred, CrystalCompAge_HippoL_train_results= runRidgeReg(
    df, 'FS_L_Hippo_Vol', ['Gender','CogCrystalComp_AgeAdj','FS_IntraCranial_Vol'])
print(CrystalCompAge_HippoL_train_results)

FluidCompAge_HippoR_linreg, FluidCompAge_HippoR_X_train, FluidCompAge_HippoR_X_test, FluidCompAge_HippoR_y_train, FluidCompAge_HippoR_y_test, FluidCompAge_HippoR_actualvspred, FluidCompAge_HippoR_train_results= runRidgeReg(
    df, 'FS_R_Hippo_Vol', ['Gender','CogFluidComp_AgeAdj','FS_IntraCranial_Vol'])
print(FluidCompAge_HippoR_train_results)

TotCompAge_HippoR_linreg, TotCompAge_HippoR_X_train, TotCompAge_HippoR_X_test, TotCompAge_HippoR_y_train, TotCompAge_HippoR_y_test, TotCompAge_HippoR_actualvspred, TotCompAge_HippoR_train_results= runRidgeReg(
    df, 'FS_R_Hippo_Vol', ['Gender','CogTotalComp_AgeAdj','FS_IntraCranial_Vol'])
print(TotCompAge_HippoR_train_results)

CrystalCompAge_HippoR_linreg, CrystalCompAge_HippoR_X_train, CrystalCompAge_HippoR_X_test, CrystalCompAge_HippoR_y_train, CrystalCompAge_HippoR_y_test, CrystalCompAge_HippoR_actualvspred, CrystalCompAge_HippoR_train_results= runRidgeReg(
    df, 'FS_R_Hippo_Vol', ['Gender','CogCrystalComp_AgeAdj','FS_IntraCranial_Vol'])
print(CrystalCompAge_HippoR_train_results)




#Lasso for cogntive components predicting volume - didn't work
#optimizeLasso(df, 'FS_L_Hippo_Vol', (['Gender','FS_IntraCranial_Vol']+AllCogAge))
#optimizeLasso(df, 'FS_R_Hippo_Vol', (['Gender','FS_IntraCranial_Vol']+AllCogAge))

#optimizeLasso(df, 'FS_L_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])
#optimizeLasso(df, 'FS_R_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])


Cog2Age_HippoL_linreg, Cog2Age_HippoL_X_train, Cog2Age_HippoL_X_test, Cog2Age_HippoL_y_train, Cog2Age_HippoL_y_test, Cog2Age_HippoL_actualvspred, Cog2Age_HippoL_train_results, Cog2Age_HippoL_Alpha = optimizeRidgeFull(
    df, 'FS_L_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])
print(Cog2Age_HippoL_train_results)



Cog2Age_HippoR_linreg, Cog2Age_HippoR_X_train, Cog2Age_HippoR_X_test, Cog2Age_HippoR_y_train, Cog2Age_HippoR_y_test, Cog2Age_HippoR_actualvspred, Cog2Age_HippoR_train_results= runRidgeReg(
    df, 'FS_R_Hippo_Vol', ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'])
print(Cog2Age_HippoR_train_results)


X, y = splitdata_normalize(df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
n_alphas = 200
#alphas = np.logspace(-10, -2, n_alphas)
alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

# #############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
#ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()




# #############################################################################


model = Ridge()
visualizer = ResidualsPlot(model, hist=False, qqplot=True)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()








