import pandas as pd
import numpy as np
from numpy import arange
from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn import metrics
import regressors
from pandas import read_csv
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot, alphas, ManualAlphaSelection











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



def checkOptimalAlpha(X, y):
    # Create a list of alphas to cross-validate against
    alphas = np.logspace(-10, 1, 400)

    # Instantiate the linear model and visualizer
    visualizer = ManualAlphaSelection(
        Ridge(),
        alphas=alphas,
        cv=12,
        scoring="neg_mean_squared_error"
    )

    visualizer.fit(X, y)
    visualizer.show()
    

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

def runRidgeReg(df, feat, predictors, alpha=1):
    X, y = splitdata_normalize(df, feat, predictors)

    # Code from try1
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)
    
    alpha = checkAlpha(X_train, X_test, y_train, y_test)

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



X, y = splitdata_normalize(df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars)
checkOptimalAlpha(X,y)
X,y = splitdata_normalize(df, 'FS_R_Hippo_Vol', pred_PSQI_tot_vars)
checkOptimalAlpha(X,y)
X,y = splitdata_normalize(df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars)
checkOptimalAlpha(X,y)
X,y = splitdata_normalize(df, 'FS_R_Hippo_Vol', pred_PSQI_comp_vars)
checkOptimalAlpha(X,y)














