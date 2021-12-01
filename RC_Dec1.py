#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#Rachel's code December 1st

#import data
df = pd.read_excel('CondensedDataandKey.xlsx',sheet_name='KeyAll')


#ignore this but don't delete - just lists of variabls in case I want to go back and add in
#brain_vars = ['FS_L_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_L_AccumbensArea_Vol', 'FS_R_Hippo_Vol', 'FS_R_Amygdala_Vol', 'FS_R_AccumbensArea_Vol']


#data cleanup
#make gender binary numbers
df['Gender'] = [0 if i == 'M' else 1 for i in df['Gender']]

# The fluid cognition composite: Dimensional Change Card Sort, Flanker, Picture Sequence Memory, List Sorting and Pattern Comparison. 
# The crystalized cognition composite: Picture Vocabulary Test and the Oral Reading Recognition Test. 

main_vars = ['Subject', 'Gender', 'Age', 'FS_IntraCranial_Vol', 'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol']
all_PSQI_vars =['PSQI_Score', 'PSQI_Comp1', 'PSQI_Comp2', 'PSQI_Comp3', 'PSQI_Comp4', 'PSQI_Comp5', 'PSQI_Comp6', 'PSQI_Comp7', 'PSQI_BedTime', 'PSQI_Min2Asleep', 'PSQI_GetUpTime', 'PSQI_AmtSleep', 'PSQI_Latency30Min', 'PSQI_WakeUp', 'PSQI_Bathroom', 'PSQI_Breathe', 'PSQI_Snore', 'PSQI_TooCold', 'PSQI_TooHot', 'PSQI_BadDream', 'PSQI_Pain', 'PSQI_Other', 'PSQI_Quality', 'PSQI_SleepMeds', 'PSQI_DayStayAwake', 'PSQI_DayEnthusiasm', 'PSQI_BedPtnrRmate']
all_NIH_CogBat_vars = ['PicSeq_Unadj', 'PicSeq_AgeAdj', 'CardSort_Unadj', 'CardSort_AgeAdj', 'Flanker_Unadj', 'Flanker_AgeAdj', 'ReadEng_Unadj', 'ReadEng_AgeAdj', 'PicVocab_Unadj', 'PicVocab_AgeAdj', 'ProcSpeed_Unadj', 'ProcSpeed_AgeAdj', 'ListSort_Unadj', 'ListSort_AgeAdj', 'CogFluidComp_Unadj', 'CogFluidComp_AgeAdj', 'CogEarlyComp_Unadj', 'CogEarlyComp_AgeAdj', 'CogTotalComp_Unadj', 'CogTotalComp_AgeAdj', 'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj']


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
