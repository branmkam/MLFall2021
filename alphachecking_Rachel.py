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
from yellowbrick.regressor import ResidualsPlot,  AlphaSelection
import matplotlib.pyplot as plt

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

def runRidgeRegPresplit(df, X_train, X_test, y_train, y_test, alpha=1):
    linreg = Ridge(fit_intercept=True, alpha=alpha)
    linreg.fit(X_train, y_train)
    #train_results = getcoeffandpvals(linreg, X_train, y_train)
    coefs = [linreg.intercept_] + list(linreg.coef_)
    train_results = pd.DataFrame(coefs, (['intercept']+X_train.columns.to_list()), columns=['coeff'])

    y_pred = linreg.predict(X_test)
    actualvspred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
    print(actualvspred.sort_values(by='Difference', ascending=False))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Alpha:', alpha)
    return linreg, actualvspred, train_results

def checkOptimalAlphaTrain(X_train, y_train, savefigure=False):
    alphas = np.logspace(-10, 1.4, 400)

    # Instantiate the linear model and visualizer
    model = RidgeCV(alphas=alphas)
    visualizer = AlphaSelection(model)
    visualizer.fit(X_train, y_train)
    opt_alpha = (visualizer.alpha_)
    ax = visualizer.show()
    if savefigure != False:
        ax.figure.savefig(savefigure)
    return opt_alpha

def barPlotRegCoef(model, predictors, savefigure=False):
    coef = pd.Series(model.coef_.flatten(), predictors).sort_values()
    plt.figure(figsize=(10,8))
 
    coef.plot(kind='bar', title='Model Coefficients')
    if savefigure != False:
        plt.savefig(savefigure)
        plt.show()
    else: 
        plt.show()
        
def regressionPlot(model, savefigure=False):
    visualizer = ResidualsPlot(model)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    ax = visualizer.show()
    if savefigure != False:
        ax.figure.savefig(savefigure)



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


df['Bilat_Hippo_Vol'] = df['FS_L_Hippo_Vol'] + df['FS_R_Hippo_Vol']

ICV, HippoVol = splitdata(df, 'FS_IntraCranial_Vol', 'Bilat_Hippo_Vol')
ICV = pd.DataFrame(ICV)
normlize_hippo = LinearRegression()
normlize_hippo.fit(ICV, HippoVol)

prediction = normlize_hippo.predict(ICV)
df['Bilat_Hippo_normICV'] = (HippoVol - prediction)

#variables we're controlling for
control_vars = ['Gender','AgeCat']

#Total PSQI and controls
pred_PSQI_tot_vars = ['PSQI_Score', 'Gender','AgeCat']

#PSQI Component subscores (7) 
pred_PSQI_comp_vars = ['PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'] + control_vars

pred_cog_vars = ['CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj'] + control_vars


print('\nModel 1: Predicting HV using Global PSQI \n')
X_1, y_1 = splitdata_normalize(df, 'Bilat_Hippo_normICV', pred_PSQI_tot_vars)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.25, random_state=None)
alpha_1 = checkOptimalAlphaTrain(X_train_1, y_train_1, savefigure = 'alphaTotPSQI.png')
TotPSQI_reg, TotPSQI_actualvspred, TotPSQI_train_results= runRidgeRegPresplit(
    df, X_train_1, X_test_1, y_train_1, y_test_1, alpha=alpha_1)

barPlotRegCoef(TotPSQI_reg, ['Global PSQI', 'Gender', 'Age'],savefigure='Model1.png')


print('\nModel 2: Predicting HV using PSQI Components \n')

X_2, y_2 = splitdata_normalize(df, 'Bilat_Hippo_normICV', pred_PSQI_comp_vars)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.25, random_state=None)
alpha_2 = checkOptimalAlphaTrain(X_train_2, y_train_2, savefigure = 'alphaCompPSQI.png')
CompPSQI_reg, CompPSQI_actualvspred, CompPSQI_train_results= runRidgeRegPresplit(
    df, X_train_2, X_test_2, y_train_2, y_test_2, alpha=alpha_2)

barPlotRegCoef(CompPSQI_reg, ['PSQI-1', 'PSQI-2', 'PSQI-3', 'PSQI-4', 'PSQI-5', 'PSQI-6', 'PSQI-7', 'Gender', 'Age'],savefigure='Model2.png')


print('\nModel 3: Predicting HV using Cognitive Scores \n')

X_3, y_3 = splitdata_normalize(df, 'Bilat_Hippo_normICV', pred_cog_vars)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.25, random_state=None)
alpha_3 = checkOptimalAlphaTrain(X_train_3, y_train_3, savefigure = 'alphaCompIntelligence.png')
Cog2_reg, Cog2_actualvspred, Cog2_train_results= runRidgeRegPresplit(
    df, X_train_3, X_test_3, y_train_3, y_test_3, alpha=alpha_3)

barPlotRegCoef(Cog2_reg, ['Fluid Cognition', 'Crystallized Cognition', 'Gender', 'Age'],savefigure='Model3.png')


print('\nModel 4: Predicting HV using Sleep + Cognitive Scores \n')
X, y = splitdata_normalize(df, 'Bilat_Hippo_Vol', ['CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7'] + control_vars)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)
alpha = checkOptimalAlphaTrain(X_train, y_train, savefigure = 'alphaModel4.png')
All_reg, All_actualvspred, All_train_results= runRidgeRegPresplit(
    df, X_train, X_test, y_train, y_test, alpha=alpha)


barPlotRegCoef(All_reg, ['Fluid Cognition', 'Crystallized Cognition', 'PSQI-1', 'PSQI-2', 'PSQI-3', 'PSQI-4', 'PSQI-5', 'PSQI-6', 'PSQI-7', 'Gender', 'Age'],savefigure='Model4.png')





