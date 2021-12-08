import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#from plotly.offline import plot
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go
from statsmodels.graphics.api import abline_plot
import statsmodels.robust.norms as rnorms

from statsmodels.compat.python import lrange, lzip
#from statsmodels.compat.pandas import Appender

import numpy as np
import pandas as pd
from patsy import dmatrix

from statsmodels.regression.linear_model import OLS, GLS, WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tools.tools import maybe_unwrap_results


os.chdir('/Users/rcorr/Documents/Belger/Manuscript/PythonMay')

#from FinalAnalysis03272020 import splitdfmeans

def ScatterPlotwRegLine(df,xvar,yvar,figuresize=(8, 8),dotcolorname='dodgerblue',linecolorname='mediumblue',savefigure=False):
    columns = [xvar,yvar]
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()
    x = df2[xvar] ## X usually means our input variables (or independent variables)
    y = df2[yvar]
    fig, ax = plt.subplots(figsize=figuresize)
    ax.scatter(x, y, alpha=0.5, color=dotcolorname)
    x = sm.add_constant(x) # constant intercept term
    model = sm.OLS(y, x, hasconst=True)
    fitted = model.fit()
    xminmax = np.linspace(x.min(), x.max(), 2)
    y_pred = fitted.predict(xminmax)
    ax.plot([x[xvar].min(),x[xvar].max()], y_pred, '-', color=linecolorname, linewidth=2)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    if savefigure != False:
        fig.savefig(savefigure)

def ScatterPlotwGLMLine(df,xvar,yvar,figuresize=(8, 8),dotcolorname='dodgerblue',linecolorname='mediumblue',savefigure=False):
    columns = [xvar,yvar]
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()
    x = df2[xvar] ## X usually means our input variables (or independent variables)
    y = df2[yvar]
    fig, ax = plt.subplots(figsize=figuresize)
    ax.scatter(x, y, alpha=0.5, color=dotcolorname)
    x = sm.add_constant(x) # constant intercept term
    model = sm.GLM(y, x)
    fitted = model.fit()
    xminmax = np.linspace(x.min(), x.max(), 2)
    y_pred = fitted.predict(xminmax)
    ax.plot([x[xvar].min(),x[xvar].max()], y_pred, '-', color=linecolorname, linewidth=2)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    if savefigure != False:
        fig.savefig(savefigure)    

def ScatterPlotwGLMLineModel(df,xvar,yvar,othervar,figuresize=(8, 8),dotcolorname='dodgerblue',linecolorname='mediumblue',savefigure=False):
    columns = [xvar,yvar]+othervar
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()    
    #creates hours after midnight linear model and finds residuals, then identifies responder/non-responder    
    x = df2[[xvar]+othervar] ## X usually means our input variables (or independent variables)
    y = df2[yvar]   
    x = sm.add_constant(x) # constant intercept term     
    model = sm.GLM(y, x).fit()
    ypred = model.predict(x)
    ypred.name = yvar + 'Resid'
    fig, ax = plt.subplots(figsize=figuresize)
    ax.scatter(x[xvar], ypred, alpha=0.5, color=dotcolorname)
    #xminmax = np.linspace(x[xvar].min(), x[xvar].max(), 2)
    #yline = model.predict(xminmax)
    #ax.plot([x[xvar].min(),x[xvar].max()], yline, '-', color=linecolorname, linewidth=2)
    #abline_plot(model_results=model, ax=ax)
    fig = abline_plot(0, model.params[0], color='k', ax=ax)
    ax.set_xlabel(xvar)
    ax.set_ylabel(ypred.name)
    if savefigure != False:
        fig.savefig(savefigure)    

def ScatterPlotwRLMLine(df,xvar,yvar,othervar,HC="H1",normopt="HuberT",scale='mad',figuresize=(8, 8),dotcolorname='dodgerblue',linecolorname='mediumblue',savefigure=False):
    columns = [xvar,yvar]+othervar
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()    
    x = df2[[xvar]+othervar] ## X usually means our input variables (or independent variables)
    y = df2[yvar]   
    X = sm.add_constant(x) # constant intercept term     
    model = sm.RLM(y, X, getattr(rnorms, normopt)()).fit(cov=HC,scale_est='mad')
    ypred = model.predict(X)
    ypred.name = yvar + 'Resid'
    fig, ax = plt.subplots(figsize=figuresize)
    ax.scatter(x[xvar], ypred, alpha=0.5, color=dotcolorname)
    #xminmax = np.linspace(x[xvar].min(), x[xvar].max(), 2)
    #yline = model.predict(xminmax)
    #ax.plot([x[xvar].min(),x[xvar].max()], yline, '-', color=linecolorname, linewidth=2)
    #abline_plot(model_results=model, ax=ax)
    #fig = abline_plot(0, model.params[0], color='k', ax=ax)
    #params = model.params
    #fig = abline_plot(*params, **dict(ax=ax,color=linecolorname))
    fig = abline_plot(intercept=model.params['const'], slope=model.params[xvar], color='k', ax=ax)

    ax.set_xlabel(xvar)
    ax.set_ylabel(ypred.name)
    if savefigure != False:
        fig.savefig(savefigure) 


def ScatterPlotwRLMLineResid(df,xvar,yvar,othervar,HC="H1",normopt="HuberT",scale='mad',figuresize=(8, 8),dotcolorname='dodgerblue',linecolorname='mediumblue',savefigure=False):
    columns = [xvar,yvar]+othervar
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()    
    x = df2[[xvar]+othervar] ## X usually means our input variables (or independent variables)
    y = df2[yvar]   
    x = sm.add_constant(x) # constant intercept term     
    model = sm.RLM(y, x).fit()
    ypred = model.predict(x)
    ypred.name = yvar + 'Resid'
    fig, ax = plt.subplots(figsize=figuresize)
    ax.scatter(x[xvar], ypred, alpha=0.5, color=dotcolorname)
    #xminmax = np.linspace(x[xvar].min(), x[xvar].max(), 2)
    #yline = model.predict(xminmax)
    #ax.plot([x[xvar].min(),x[xvar].max()], yline, '-', color=linecolorname, linewidth=2)
    #abline_plot(model_results=model, ax=ax)
    fig = abline_plot(0, model.params[0], color='k', ax=ax)
    ax.set_xlabel(xvar)
    ax.set_ylabel(ypred.name)
    if savefigure != False:
        fig.savefig(savefigure) 
        
def runRLM(df,listxvars,yvar,HC="H1",normopt="HuberT",scale='mad'):
    if yvar in listxvars:
        listxvars = listxvars.remove(yvar)
    columns = (listxvars+[yvar])
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()
    X = df2[listxvars] ## X usually means our input variables (or independent variables)
    y = df2[yvar] ## Y usually means our output/dependent variable
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    #results = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit(cov=HC) 
    if scale == 'Huber2':
        results = sm.RLM(y, X, getattr(rnorms, normopt)()).fit(cov=HC,scale_est=sm.robust.scale.HuberScale())
    else:
        results = sm.RLM(y, X, getattr(rnorms, normopt)()).fit(cov=HC,scale_est='mad')
    n = (len(y))
    #predictions = model.predict(X)
    coefs = []
    dfcoefs = results.params
    for num in range(0,len(dfcoefs)):
        coefs.append(((dfcoefs.index[num]),(format((dfcoefs[num]),'.4g'))))
    stderr = []
    dfbse = results.bse
    for num in range(0,len(dfbse)):
        stderr.append(((dfbse.index[num]),(format((dfbse[num]),'.4g'))))
    pvals = []
    pflag = []
    dfpvals = results.pvalues
    for num in range(0,len(dfpvals)):
        pvals.append(((dfpvals.index[num]),(format((dfpvals[num]),'.4g'))))
        if dfpvals.index[num] != 'const' and dfpvals[num] < 0.05:
            if dfpvals[num] >= 0.001:
                pflag.append(str(dfpvals.index[num]) + ' p=' + str(round(dfpvals[num],3)))
            else:
                pflag.append(str(dfpvals.index[num]) + ' p<0.001')
    if len(pflag) == 0:
        pflag = ['None Sig']
    tvals = []
    dftvals = results.tvalues
    for num in range(0,len(dfpvals)):
        tvals.append(((dftvals.index[num]),(format((dftvals[num]),'.4g'))))
    dict = {'coefs': str(coefs), 'stderror': str(stderr), 'tvalues': str(tvals), 'pvalues': str(pvals), 'N': str(n), 'OLSmodel': results,'summary': results.summary(),'pflag': ' '.join(pflag)}  
    resultsdf = pd.DataFrame(dict,index=[yvar + ' vs ' + "+".join(listxvars)]) 
    resultsdf = resultsdf.reindex(columns=['coefs','stderror','tvalues','pvalues','pflag','N','summary','OLSmodel'])
    return(resultsdf)

def plot_ccpr_RLM(df,listxvars,yvar,sigvar,HC="H1",normopt="TrimmedMean",scale='mad',figuresize=(8, 8),savefigure=False, plotoutlier=True, cropgraph=False):
    columns = (listxvars+[yvar])
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()
    X = df2[listxvars] ## X usually means our input variables (or independent variables)
    y = df2[yvar] ## Y usually means our output/dependent variable
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    results = sm.RLM(y, X, getattr(rnorms, normopt)()).fit(cov=HC,scale_est='mad')
    
    fig, ax = plt.subplots(figsize=figuresize)
    plot_ccpr5(results, sigvar, yvar, ax=ax,plotoutlier=plotoutlier,cropgraph=cropgraph)
    if savefigure != False:
        fig.savefig(savefigure) 
    return fig

def plot_ccpr_RLM_TM(df,listxvars,yvar,sigvar,HC="H1",cval=2,figuresize=(8, 8),savefigure=False, plotoutlier=True, cropgraph=False):
    columns = (listxvars+[yvar])
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()
    X = df2[listxvars] ## X usually means our input variables (or independent variables)
    y = df2[yvar] ## Y usually means our output/dependent variable
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    results = sm.RLM(y, X, M=sm.robust.norms.TrimmedMean(c=cval)).fit(cov=HC,scale_est='mad')
    fig, ax = plt.subplots(figsize=figuresize)
    plot_ccpr5(results, sigvar, yvar, ax=ax,plotoutlier=plotoutlier,cropgraph=cropgraph)
    if savefigure != False:
        fig.savefig(savefigure) 
    return fig

def plot_ccpr_RLM_Huber(df,listxvars,yvar,sigvar,HC="H1",cval=2,figuresize=(8, 8),savefigure=False, plotoutlier=True, cropgraph=False):
    columns = (listxvars+[yvar])
    df2 = df.loc[:, columns] 
    df2 = df2.dropna()
    X = df2[listxvars] ## X usually means our input variables (or independent variables)
    y = df2[yvar] ## Y usually means our output/dependent variable
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    results = sm.RLM(y, X, M=sm.robust.norms.TrimmedMean(c=cval)).fit(cov=HC,scale_est=sm.robust.scale.HuberScale())
    fig, ax = plt.subplots(figsize=figuresize)
    plot_ccpr5(results, sigvar, yvar, ax=ax,plotoutlier=plotoutlier,cropgraph=cropgraph)
    if savefigure != False:
        fig.savefig(savefigure) 
    return fig


def plot_ccpr5(results, exog_idx, yvar, ax=None, dotcolor='blue', linecolor='blue',plotoutlier=True,cropgraph=False):
    """
    Plot CCPR against one regressor.

    Generates a component and component-plus-residual (CCPR) plot.

    Parameters
    ----------
    results : result instance
        A regression results instance.
    exog_idx : {int, str}
        Exogenous, explanatory variable. If string is given, it should
        be the variable name that you want to use, and you can use arbitrary
        translations as with a formula.
    ax : AxesSubplot, optional
        If given, it is used to plot in instead of a new figure being
        created.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    plot_ccpr_grid : Creates CCPR plot for multiple regressors in a plot grid.

    Notes
    -----
    The CCPR plot provides a way to judge the effect of one regressor on the
    response variable by taking into account the effects of the other
    independent variables. The partial residuals plot is defined as
    Residuals + B_i*X_i versus X_i. The component adds the B_i*X_i versus
    X_i to show where the fitted line would lie. Care should be taken if X_i
    is highly correlated with any of the other independent variables. If this
    is the case, the variance evident in the plot will be an underestimate of
    the true variance.

    References
    ----------
    http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ccpr.htm

    Examples
    --------
    Using the state crime dataset plot the effect of the rate of single
    households ('single') on the murder rate while accounting for high school
    graduation rate ('hs_grad'), percentage of people in an urban area, and rate
    of poverty ('poverty').

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plot
    >>> import statsmodels.formula.api as smf

    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> sm.graphics.plot_ccpr(results, 'single')
    >>> plt.show()

    .. plot:: plots/graphics_regression_ccpr.py
    """
    fig2, ax = utils.create_mpl_ax(ax)

    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    results = maybe_unwrap_results(results)

    x1 = results.model.exog[:, exog_idx]
    #namestr = ' for %s' % self.name if self.name else ''
    x1beta = x1*results.params[exog_idx]
    #ax.plot(x1, x1beta + results.resid, 'o',color=dotcolor,alpha=0.3)
    bool_val = list(map(int, results.weights))
    xdf = pd.DataFrame(data=[x1,(x1beta+results.resid),bool_val],index=['x1','x1beta+resid','outliers']).T
    groups = xdf.groupby("outliers")
    if plotoutlier==True:
        ax.plot(groups.get_group(0)["x1"], groups.get_group(0)["x1beta+resid"], marker="x", color="red", linestyle="", label="Outliers",alpha=0.5)
    ax.plot(groups.get_group(1)["x1"], groups.get_group(1)["x1beta+resid"], marker="o", color="blue", linestyle="", label="Data",alpha=0.5)
    from statsmodels.tools.tools import add_constant
    #mod = OLS(x1beta, add_constant(x1)).fit()
    #sm.RLM(y, X, getattr(rnorms, normopt)()).fit(cov=HC,scale_est='mad')
    #mod = sm.RLM(x1beta, add_constant(x1),getattr(rnorms, normopt)()).fit(cov=HC,scale_est='mad')
    mod = sm.WLS(x1beta, add_constant(x1),weights=results.weights).fit(cov=HC,scale_est='mad')
    params = mod.params
    fig2 = abline_plot(*params, **dict(ax=ax,color=linecolor))
    #sm.graphics.plot_ccpr(results, sigvar, ax=ax)
    #ax.plot(x1, x1beta, '-')
    #ax.set_title((yvar + ' vs ' + exog_name + ' Component and Component Plus Residual Plot'))
    ax.set_title(((yvar[6:]).replace("_"," (") + ")") + ' vs ' + exog_name) # + ' Component and Component Plus Residual Plot'))
    #ax.set_ylabel("%s Residual + %s*beta_%d" % (yvar, exog_name, exog_idx))
    ax.set_ylabel("%s Mean Beta Residual" % ((yvar[6:]).replace("_"," (") + ")"))
    ax.set_xlabel("%s" % exog_name)
#    fig2.update_layout(
#            title=(((yvar[6:-4]).replace("_"," (") + ")") + ' vs ' + exog_name),
#            xaxis_title=("%s" % exog_name),
#            yaxis_title=("%s Residual" % ((yvar[6:-4]).replace("_"," (") + ")")),
#            font=dict(
#                    family="Arial",
#                    size=18,
#                    color="#7f7f7f"
#                    )
#            )
    plt.rcParams.update({'font.size': 18})
    if cropgraph==True:
        ax.set_ylim((min(groups.get_group(1)["x1beta+resid"])-2),(max(groups.get_group(1)["x1beta+resid"])+2))
    return fig2

def splitbargraph(df,listxvars,yvar,groupvar,var0name='Var0',var1name='Var1',groupvarname=False,color0='#4269f5',color1='#eb0c0c',savefigure=False):
    df0 = df.loc[df[groupvar] == 0]
    df1 = df.loc[df[groupvar] == 1]
    y0 = df0[yvar].tolist()
    y1 = df1[yvar].tolist()
    x0 = []
    x0.extend(([var0name] * (len(y0))))
    x1 = []
    x1.extend(([var1name] * (len(y1))))
    hoversubject0=[]    
    hoversubject0.extend(df0.index.to_list())
    hoversubject1=[]
    hoversubject1.extend(df.index.to_list())

    trace0 = go.Box(y=y0,x=x0,name=var0name, hovertext=hoversubject0,marker=dict(color=color0))
    trace1 = go.Box(y=y1,x=x1,name=var1name, hovertext=hoversubject1,marker=dict(color=color1))
    data = [trace0, trace1]
    layout = go.Layout(
            yaxis=dict(title=((yvar[6:]).replace("_"," (") + ")" + ' Mean Beta'), zeroline=True,zerolinecolor='rgb(0, 0, 0)',zerolinewidth=2),
            boxmode='overlay', 
            font=dict(family="Arial",size=20,color="#000000"),
            boxgap=0.1,boxgroupgap=0,
            width=550, height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(data=data, layout=layout)
    if groupvarname != False:
        fig.update_layout(title=(groupvarname + ' ' + (yvar[6:]).replace("_"," (") + ")"),boxmode='overlay', showlegend=False,title_x=0.5)
    else:
        fig.update_layout(title=(groupvar + ' vs ' + (yvar[6:]).replace("_"," (") + ")"),boxmode='overlay', showlegend=False,title_x=0.5)
    plot(fig)
    if savefigure != False:
        fig.write_image(savefigure)


os.chdir('/Users/rcorr/Documents/Belger/Manuscript/PythonMay')

df = pd.read_excel("Output05122020_S1S6Corrected_NoHRV_TM_H3_mad.xlsx",sheet_name="df")

subjectsmissingdata = ['cnt_129_bl','cnt_182_bl','cnt_183_bl','cnt_237_bl']
df = df.loc[~df['Subject'].isin(subjectsmissingdata)]

df = df.set_index('Subject')

listxvars = ['Sex','Age','CortResponder','STAIT']

os.chdir('/Users/rcorr/Documents/Belger/Manuscript/PythonMay/Graphs')


ContDict = dict([('Cope4_Hippocampus_L', 'STAIT'),('Cope4_VS_L', 'STAIT'),('Cope4_Precuneus_L', 'Age')]) 

for yvar, sigvar in ContDict.items():
    #plot_ccpr_RLM(df,listxvars,yvar,sigvar,HC="H2",normopt="TrimmedMean",scale='mad',figuresize=(8, 8),plotoutlier=True,cropgraph=False,savefigure=(yvar[6:]+'_'+sigvar+'.png'))
    plot_ccpr_RLM(df,listxvars,yvar,sigvar,HC="H3",normopt="TrimmedMean",scale='mad',figuresize=(8, 8),plotoutlier=True,cropgraph=False,savefigure=(yvar[6:]+'_'+sigvar+'.png'))
    plot_ccpr_RLM(df,listxvars,yvar,sigvar,HC="H3",normopt="TrimmedMean",scale='mad',figuresize=(8, 8),plotoutlier=False,cropgraph=False,savefigure=(yvar[6:]+'_'+sigvar+'_NoOutlier.png'))

splitbargraph(df,listxvars,'Cope4_dlPFC_R','CortResponder',var0name='Non-Responder',var1name='Responder',groupvarname = 'Cortisol Responders',color0='#91e1d6',color1="#04b050",savefigure='dlPFC_R_Cort.png')
splitbargraph(df,listxvars,'Cope4_Putamen_R','Sex',var0name='Non-Responder',var1name='Responder',groupvarname = 'Cortisol Responders',color0='#91e1d6',color1="#04b050",savefigure='Precuneus_L_Cort.png')
              
df_dlPFC = df.loc[df['ModelUsing_dlPFC_R']==1]
df_Putamen = df.loc[df['ModelUsing_Putamen_R']==1]

splitbargraph(df_dlPFC,listxvars,'Cope4_dlPFC_R','CortResponder',var0name='Non-Responder',var1name='Responder',groupvarname = 'Cortisol Responders',color0='#91e1d6',color1="#04b050",savefigure='dlPFC_R_Cort_NoOutlier.png')
splitbargraph(df_Putamen,listxvars,'Cope4_Putamen_R','Sex',var0name='Non-Responder',var1name='Responder',groupvarname = 'Cortisol Responders',color0='#91e1d6',color1="#04b050",savefigure='Precuneus_L_Cort_NoOutlier.png')

plot_ccpr_RLM_TM(df,listxvars,'Cope4_Hippocampus_L','STAIT',HC="H3",cval=2.3,figuresize=(8, 8),plotoutlier=True,cropgraph=False,savefigure=('Hippocampus_L_STAIT_cval23.png'))
plot_ccpr_RLM_TM(df,listxvars,'Cope4_Hippocampus_L','STAIT',HC="H3",cval=2.3,figuresize=(8, 8),plotoutlier=False,cropgraph=False,savefigure=('Hippocampus_L_STAIT_cval23_NoOutlier.png'))
