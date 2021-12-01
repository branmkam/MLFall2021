@@ -0,0 +1,147 @@
#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import time


#import data
df = pd.read_csv('CondensedDataandKey.csv')


#data cleanup
#make gender binary numbers
df['Gender'] = [0 if i == 'M' else 1 for i in df['Gender']]

#take some columns - can paste directly from CSV and delete what needed here
cols = 'Gender,MMSE_Score,PSQI_Min2Asleep,PSQI_AmtSleep,PSQI_Latency30Min,PSQI_WakeUp,PSQI_Bathroom,PSQI_Breathe,PSQI_Snore,PSQI_TooCold,PSQI_TooHot,PSQI_BadDream,PSQI_Pain,PSQI_Other,PSQI_Quality,PSQI_SleepMeds,PSQI_DayStayAwake,PSQI_DayEnthusiasm,PSQI_BedPtnrRmate,PicSeq_Unadj,PicSeq_AgeAdj,CardSort_Unadj,CardSort_AgeAdj,Flanker_Unadj,Flanker_AgeAdj,ReadEng_Unadj,ReadEng_AgeAdj,PicVocab_Unadj,PicVocab_AgeAdj,ProcSpeed_Unadj,ProcSpeed_AgeAdj,ListSort_Unadj,ListSort_AgeAdj,CogFluidComp_Unadj,CogFluidComp_AgeAdj,CogEarlyComp_Unadj,CogEarlyComp_AgeAdj,CogTotalComp_Unadj,CogTotalComp_AgeAdj,CogCrystalComp_Unadj,CogCrystalComp_AgeAdj,ER40_CR,ER40_CRT,ER40ANG,ER40FEAR,ER40HAP,ER40NOE,ER40SAD,AngAffect_Unadj,AngHostil_Unadj,AngAggr_Unadj,FearAffect_Unadj,FearSomat_Unadj,Sadness_Unadj,LifeSatisf_Unadj,MeanPurp_Unadj,PosAffect_Unadj,Friendship_Unadj,Loneliness_Unadj,PercHostil_Unadj,PercReject_Unadj,EmotSupp_Unadj,InstruSupp_Unadj,PercStress_Unadj,SelfEff_Unadj,FS_L_Hippo_Vol,FS_L_Amygdala_Vol,FS_L_AccumbensArea_Vol,FS_R_Hippo_Vol,FS_R_Amygdala_Vol,FS_R_AccumbensArea_Vol,WM_Task_Acc,WM_Task_Median_RT,WM_Task_2bk_Acc,WM_Task_2bk_Median_RT,WM_Task_0bk_Acc,WM_Task_0bk_Median_RT,WM_Task_0bk_Body_Acc,WM_Task_0bk_Body_Acc_Target,WM_Task_0bk_Body_Acc_Nontarget,WM_Task_0bk_Face_Acc,WM_Task_0bk_Face_Acc_Target,WM_Task_0bk_Face_ACC_Nontarget,WM_Task_0bk_Place_Acc,WM_Task_0bk_Place_Acc_Target,WM_Task_0bk_Place_Acc_Nontarget,WM_Task_0bk_Tool_Acc,WM_Task_0bk_Tool_Acc_Target,WM_Task_0bk_Tool_Acc_Nontarget,WM_Task_2bk_Body_Acc,WM_Task_2bk_Body_Acc_Target,WM_Task_2bk_Body_Acc_Nontarget,WM_Task_2bk_Face_Acc,WM_Task_2bk_Face_Acc_Target,WM_Task_2bk_Face_Acc_Nontarget,WM_Task_2bk_Place_Acc,WM_Task_2bk_Place_Acc_Target,WM_Task_2bk_Place_Acc_Nontarget,WM_Task_2bk_Tool_Acc,WM_Task_2bk_Tool_Acc_Target,WM_Task_2bk_Tool_Acc_Nontarget,WM_Task_0bk_Body_Median_RT,WM_Task_0bk_Body_Median_RT_Target,WM_Task_0bk_Body_Median_RT_Nontarget,WM_Task_0bk_Face_Median_RT,WM_Task_0bk_Face_Median_RT_Target,WM_Task_0bk_Face_Median_RT_Nontarget,WM_Task_0bk_Place_Median_RT,WM_Task_0bk_Place_Median_RT_Target,WM_Task_0bk_Place_Median_RT_Nontarget,WM_Task_0bk_Tool_Median_RT,WM_Task_0bk_Tool_Median_RT_Target,WM_Task_0bk_Tool_Median_RT_Nontarget,WM_Task_2bk_Body_Median_RT,WM_Task_2bk_Body_Median_RT_Target,WM_Task_2bk_Body_Median_RT_Nontarget,WM_Task_2bk_Face_Median_RT,WM_Task_2bk_Face_Median_RT_Target,WM_Task_2bk_Face_Median_RT_Nontarget,WM_Task_2bk_Place_Median_RT,WM_Task_2bk_Place_Median_RT_Target,WM_Task_2bk_Place_Median_RT_Nontarget,WM_Task_2bk_Tool_Median_RT,WM_Task_2bk_Tool_Median_RT_Target,WM_Task_2bk_Tool_Median_RT_Nontarget'
colarr = cols.split(',')

#delete all unadjusted and WM_task, age not important here
adjcols = []
for i in colarr:
    if not 'Unadj' in i and not 'WM_Task' in i:
        adjcols.append(i)
        
        
        
        #bring data in
X = df[adjcols]

#get rid of any incomplete data
for row in X.index:
    if X.loc[row].isnull().values.any():
        #print(row)
        X = X.drop(labels=[row], axis='index')

X = X.reset_index(drop=True)


#split data into features and predictor
def splitdata(feat, X):   
    y = X[feat]
    X = X.drop(axis = 'columns', labels=[feat])
    return X, y


# EDIT THIS VALUE TO SELECT PREDICTED FEATURE
feat = 'PSQI_AmtSleep'
#SELECT WHAT PORTION OF DATA GETS PARTITIONED TO BE TEST DATA
test_size = 0.25

X, y = splitdata(feat, X)

#divide data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)


#add bias term as first feature
import numpy as np
X_train_pad = np.concatenate( ( np.ones((X_train.shape[0], 1)), X_train ), axis = 1 )
X_test_pad = np.concatenate( ( np.ones((X_test.shape[0], 1)), X_test ), axis = 1 )


n, p = X_train_pad.shape
print(n, 'samples to train (n)')
print(p, 'features including bias (p)')



import numpy as np
def loglikelihood(w, X, y, alpha): 
    for i in [w, X, y]:
        print(i.shape)
    #compute loglikelihood for current w, b, given the data X, y
    #w is a vector, b is a scalr, X is a n*p matrix and y is a vector.
    tmp = 1. + np.exp(-y*np.dot(X, w))
    prob = 1./tmp
    print(X.shape)
    X = X.T #X becomes a p*n matrix so the gradVal can be compute straight-forwardly.
    matrixVal = X*(np.exp(-y*np.dot(w, X))/tmp)
    gradVal = np.dot(matrixVal, y)
    penalty = (alpha/2.)*np.sum(w[1:]**2)
    gradPenalty = -alpha*(w)
    gradPenalty[0] = 0.0;
    return -np.sum( np.log( tmp ) ) - penalty, gradVal + gradPenalty

import matplotlib.pyplot as plt
%matplotlib inline
def gradient_ascent(f,x,init_step,iterations):  
    f_val,grad = f(x)                           # compute function value and gradient 
    f_vals = [f_val]
    for it in range(iterations):                # iterate for a fixed number of iterations
        #print 'iteration %d' % it
        done = False                            # initial condition for done
        line_search_it = 0                      # how many times we tried to shrink the step
        step = init_step                        # reset step size to the initial size
        while not done and line_search_it<100:  # are we done yet?
            new_x = x + step*grad               # take a step along the gradient
            new_f_val,new_grad = f(new_x)       # evaluate function value and gradient
            if new_f_val<f_val:                 # did we go too far?
                step = step*0.95                # if so, shrink the step-size
                line_search_it += 1             # how many times did we shrank the step
            else:
                done = True                     # better than the last x, so we move on
        
        if not done:                            # did not find right step size
            print("Line Search failed.")
        else:
            f_val = new_f_val                   # ah, we are ok, accept the new x
            x = new_x
            grad = new_grad
            f_vals.append(f_val)
        plt.plot(f_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    return f_val, x


#initialize w vector
w_init = np.random.randn(p)*0.01
w_init[0] = 0


def optimizeFn( init_step, iterations, alpha, w ):
    g = lambda xy0: loglikelihood(xy0, X_train_pad, y_train, alpha)
    f_val, update_w = gradient_ascent( g, w, init_step, iterations )
    return f_val, update_w
print(w_init.shape[0], 'weights:','Good' if w_init.shape[0] == p else 'PROBLEM!')

np.random.seed(1)
X = np.random.randn(2,3)
y = np.array([1,-1])
w = np.ones(3)
w[[1]] = -1;
loglikelihood(w, X, y, 1)

X = X_train_pad
y = y_train
w = w_init
loglikelihood(w, X, y, 1)


#see the error on the validation set
#fiddle with gradient step value - init_step
f_val, update_w=optimizeFn(init_step = 1e-5, iterations=100, alpha=3000, w=w_init)
