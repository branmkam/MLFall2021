from sklearn.linear_model import Ridge
#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing

def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['CogCrystalComp_AgeAdj'],y_pred)
        plt.plot(data['CogCrystalComp_AgeAdj'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

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



#https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/

alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

data=df
data['y'] = data['FS_R_Hippo_Vol']
predictors = ['Gender','FS_IntraCranial_Vol','CogFluidComp_AgeAdj','CogCrystalComp_AgeAdj']

    
#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_%s'%i for i in predictors]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)














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


features = pred_PSQI_comp_vars
X, y = splitdata(df, 'FS_L_Hippo_Vol', pred_PSQI_comp_vars)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

# loop through different penalty score (alpha) and obtain the estimated coefficient (weights)
alphas = 10 ** np.arange(1, 5)
print('different alpha values:', alphas)

# stores the weights of each feature
ridge_weight = []
for alpha in alphas:    
    ridge = Ridge(alpha = alpha, fit_intercept = True)
    ridge.fit(X_train_std, y_train)
    ridge_weight.append(ridge.coef_)
    
    
def weight_versus_alpha_plot(weight, alphas, features):
    """
    Pass in the estimated weight, the alpha value and the names
    for the features and plot the model's estimated coefficient weight 
    for different alpha values
    """
    fig = plt.figure(figsize = (8, 6))
    
    # ensure that the weight is an array
    weight = np.array(weight)
    for col in range(weight.shape[1]):
        plt.plot(alphas, weight[:, col], label = features[col])

    plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
    
    # manually specify the coordinate of the legend
    plt.legend(bbox_to_anchor = (1.3, 0.9))
    plt.title('Coefficient Weight as Alpha Grows')
    plt.ylabel('Coefficient weight')
    plt.xlabel('alpha')
    return fig

# change default figure and font size
plt.rcParams['figure.figsize'] = 8, 6 
plt.rcParams['font.size'] = 12


ridge_fig = weight_versus_alpha_plot(ridge_weight, alphas, features)

