#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#Rachel's code December 1st

#import data
df_all = pd.read_excel('CondensedDataandKey.xlsx',sheet_name='data')


#ignore this but don't delete - just lists of variabls in case I want to go back and add in
#brain_vars = ['FS_L_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_L_AccumbensArea_Vol', 'FS_R_Hippo_Vol', 'FS_R_Amygdala_Vol', 'FS_R_AccumbensArea_Vol']
#main_vars = ['Subject', 'Gender', 'Age', 'FS_IntraCranial_Vol', 'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol']
#all_PSQI_vars =['PSQI_Score', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7', 'PSQI_BedTime', 'PSQI_Min2Asleep', 'PSQI_GetUpTime', 'PSQI_AmtSleep', 'PSQI_Latency30Min', 'PSQI_WakeUp', 'PSQI_Bathroom', 'PSQI_Breathe', 'PSQI_Snore', 'PSQI_TooCold', 'PSQI_TooHot', 'PSQI_BadDream', 'PSQI_Pain', 'PSQI_Other', 'PSQI_Quality', 'PSQI_SleepMeds', 'PSQI_DayStayAwake', 'PSQI_DayEnthusiasm', 'PSQI_BedPtnrRmate']
#all_NIH_CogBat_vars = ['PicSeq_Unadj', 'PicSeq_AgeAdj', 'CardSort_Unadj', 'CardSort_AgeAdj', 'Flanker_Unadj', 'Flanker_AgeAdj', 'ReadEng_Unadj', 'ReadEng_AgeAdj', 'PicVocab_Unadj', 'PicVocab_AgeAdj', 'ProcSpeed_Unadj', 'ProcSpeed_AgeAdj', 'ListSort_Unadj', 'ListSort_AgeAdj', 'CogFluidComp_Unadj', 'CogFluidComp_AgeAdj', 'CogEarlyComp_Unadj', 'CogEarlyComp_AgeAdj', 'CogTotalComp_Unadj', 'CogTotalComp_AgeAdj', 'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj']
#Notes: 
# The fluid cognition composite: Dimensional Change Card Sort, Flanker, Picture Sequence Memory, List Sorting and Pattern Comparison. 
# The crystalized cognition composite: Picture Vocabulary Test and the Oral Reading Recognition Test. 


main_vars = ['Subject', 'Gender', 'Age', 'FS_IntraCranial_Vol', 'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol','PSQI_Score', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7','CogFluidComp_Unadj','CogFluidComp_AgeAdj','CogTotalComp_Unadj', 'CogTotalComp_AgeAdj', 'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj']
df = df_all[main_vars]
df = df.dropna()

#data cleanup
#make gender binary numbers
df['Gender'] = [0 if i == 'M' else 1 for i in df['Gender']]

age_dict = {'22-25': 1, '26-30': 2, '31-35': 3, '36+': 4}
df['AgeCat'] = df['Age'].replace(age_dict)


#split data into features and predictor
def splitdata(df, feat, predictors):   
    y = df[feat]
    X = df[predictors]
    return X, y

control_vars = ['Gender','AgeCat','FS_IntraCranial_Vol']
pred_PSQI_tot_vars = ['PSQI_Score'] + control_vars
X, y = splitdata(df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars)




# Code from try1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

#train data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Difference': y_test-y_pred})
print(df.sort_values(by='Difference', ascending=False))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))









# #Code adapted from from trymanual and https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_0_23_0.html#sphx-glr-auto-examples-release-highlights-plot-release-highlights-0-23-0-py

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import PoissonRegressor
# from sklearn.ensemble import HistGradientBoostingRegressor

# # n_samples, n_features = 1000, 20
# # rng = np.random.RandomState(0)
# # X = rng.randn(n_samples, n_features)
# # # positive integer target correlated with X[:, 5] with many zeros:
# # y = rng.poisson(lam=np.exp(X[:, 5]) / 2)

# #divide data into training and test sets
# test_size = 0.25
# control_vars = ['Gender','AgeCat','FS_IntraCranial_Vol']
# pred_PSQI_tot_vars = ['PSQI_Score'] + control_vars
# X, y = splitdata(df, 'FS_L_Hippo_Vol', pred_PSQI_tot_vars)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)


# #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
# glm = PoissonRegressor()
# gbdt = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.01)
# glm.fit(X_train, y_train)
# gbdt.fit(X_train, y_train)
# print(glm.score(X_test, y_test))
# print(gbdt.score(X_test, y_test))




#ignore this
# from sklearn import linear_model
# model = linear_model.Ridge()
# model.fit(X, y)
# model.coef_