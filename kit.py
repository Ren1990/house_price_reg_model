import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.data import tips
import warnings
warnings.filterwarnings("ignore")
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
rs= RobustScaler()
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import shap

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
#from hpsklearn import xgboost_classification
#from skopt.space import Real, Integer
#from skopt.utils import use_named_args

from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance 
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

#To run viz for all columns vs response
#for categorical target response
def viz_all2(df,target_response):
    i_col=list(df.columns)
    for i in i_col:
        if df[i].dtype !="object" and i!=target_response:
            j_col=list(df.columns)
            for j in j_col:
                if df[j].dtype=='object' and j!=target_response and j!=i:
                    plt.figure(figsize=(20,10))
                    sns.violinplot(df,x=j, y=i,split=True,hue=target_response,gap=.1, inner="quart")
                    plt.xticks(rotation=60,fontsize=8)
                    plt.title(f"Violin plot using numberic variable vs categorical variable with target response as legend:\n {i} vs {j} with {target_response} as legend")
                    j_col.remove(j)

                if df[j].dtype!='object' and j!=target_response and j!=i:
                    plt.figure(figsize=(20,10))
                    sns.scatterplot(df,x=j, y=i,hue=target_response,style=target_response)
                    plt.title(f"Scatter plot using numberic variable vs numberic variable with target response as legend:\n {i} vs {j} with {target_response} as legend")
                    j_col.remove(j)

        if df[i].dtype =="object" and i!=target_response:
            j_col=list(df.columns)
            for j in j_col:
                if df[j].dtype=='object' and j!=target_response and j!=i:
                    if len(pd.unique(df[i]))>len(pd.unique(df[j])):
                        plt.figure(figsize=(20,10))
                        plt.title(f"Histogram plot using categorical variable vs categorical variable with target response as legend:\n {i} vs {j} with {target_response} as legend")
                        sns.displot(data=df, x=i,col=target_response,hue=j)
                        plt.xticks(rotation=60,fontsize=8)
                        plt.xticks(rotation=60,fontsize=8)
                        j_col.remove(j)
                    if len(pd.unique(df[i]))<=len(pd.unique(df[j])):
                        plt.title(f"Histogram plot using categorical variable vs categorical variable with target response as legend:\n {i} vs {j} with {target_response} as legend")
                        plt.figure(figsize=(20,10))
                        sns.displot(data=df, x=j,col=target_response,hue=i)
                        plt.xticks(rotation=60,fontsize=8)
                        plt.xticks(rotation=60,fontsize=8)
                        j_col.remove(j)

                if df[j].dtype!='object' and j!=target_response and j!=i:
                    plt.figure(figsize=(20,10))
                    sns.violinplot(df,x=i, y=j,split=True,hue=target_response,gap=.1, inner="quart")
                    plt.xticks(rotation=60,fontsize=8)
                    plt.title(f"Violin plot using numberic variable vs categorical variable with target response as legend:\n {i} vs {j} with {target_response} as legend")
                    j_col.remove(j)

def viz_violinplot(df,x,y,target_response):
    fig = go.Figure()
    fig.add_trace(go.Violin(x=df[x][ df[target_response] == 'yes' ],
                            y=df[y][ df[target_response] == 'yes' ],
                            legendgroup='yes', scalegroup='yes', name='yes',
                            side='negative',
                            line_color='blue')
                )
    fig.add_trace(go.Violin(x=df[x][ df[target_response] == 'no' ],
                            y=df[y][ df[target_response] == 'no' ],
                            legendgroup='no', scalegroup='no', name='no',
                            side='positive',
                            line_color='orange')
                )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0,
                      violinmode='overlay',
                      title=f"Violin plot using numberic variable vs categorical variable with target response as legend:\n {y} vs {x} with {target_response} as legend",
                      xaxis_title=x,
                      yaxis_title=y,
                      legend_title=target_response
                      )
    fig.show()

def viz_scatter(df,col,target_response,catagory):
    fig = px.scatter(df, x=col,y=target_response,color=catagory)
    fig.update_layout(title=f'Scatter plot of {target_response} vs {col} with {catagory} as label')
    fig.show()

def viz_hist(df,i,target_response):
    fig = px.histogram(df,
                       x=i,
                        color=target_response,
                        text_auto= True
                        )
    fig.update_layout(title=f"Count distribution plot of {i} with {target_response} as legend",
                        xaxis_title=i,
                        legend_title=target_response
                        )            
    fig.show()

    fig = px.histogram(df,
                       x=i,
                        color=target_response,
                        barnorm='percent',
                        text_auto= True
                        )
    fig.update_traces(texttemplate='%{y:.2f}') 
    fig.update_layout(title=f"Density distribution plot of {i} with {target_response} as legend",
                        xaxis_title=i,
                        legend_title=target_response
                        )            
    fig.show()

#old data transform function. Apply Robust Scaler to numerical columns, and Ordinal Encoder for categorical columns
def pro_df(df_tr):
    numerical_ix = df_tr.select_dtypes(include=['int64', 'float64']).columns
    categorical_ix = df_tr.select_dtypes(include=['object', 'bool']).columns
    
    #drop numerical feat and convert cat feat
    pro_cat=df_tr
    for i in numerical_ix:
        pro_cat=pro_cat.drop(i,axis=1)
    pro_cat=pd.DataFrame(oe.fit_transform(pro_cat),columns=list(pro_cat.columns))

    #drop cat feat and convert numerical feat
    pro_num=df_tr
    for i in categorical_ix:
        pro_num=pro_num.drop(i,axis=1)
    pro_num=pd.DataFrame(rs.fit_transform(pro_num),columns=list(pro_num))

    return pd.concat([pro_cat,pro_num],axis=1)

#Latest data transform function, apply Ordinary Encoder to (binary) response, One Hot Encoder to categorical col, and Robust Scaler to numberical col
#Applying One Hot Encoder will create more sub-col, i.e. col count is increased
def pro_df2(df_tr,response):
    #convert cat response into ordinal
    resp_df=pd.DataFrame(df_tr[response])
    pro_resp=pd.DataFrame(oe.fit_transform(resp_df),columns=list(resp_df.columns))
    df_tr=df_tr.drop(response,axis=1)
    
    numerical_ix = df_tr.select_dtypes(include=['int64', 'float64']).columns
    categorical_ix = df_tr.select_dtypes(include=['object', 'bool']).columns
    
    #drop numerical feat and convert cat feat
    pro_cat=df_tr
    for i in numerical_ix:
        pro_cat=pro_cat.drop(i,axis=1)
    for i in categorical_ix:
        ohe_df=pd.DataFrame(ohe.fit_transform(pro_cat[[i]]).toarray(),columns=ohe.get_feature_names_out())
        pro_cat=pro_cat.drop(i,axis=1)
        pro_cat=pd.concat([ohe_df,pro_cat],axis=1)

    #drop cat feat and convert numerical feat
    pro_num=df_tr
    for i in categorical_ix:
        pro_num=pro_num.drop(i,axis=1)
    pro_num=pd.DataFrame(rs.fit_transform(pro_num),columns=list(pro_num))

    return pd.concat([pro_cat,pro_num,pro_resp],axis=1)

#split response col and train,eval split
def tr_val_split(df,target_response):
    x=df.drop(target_response,axis=1)
    y=df[target_response]
    return train_test_split(x,y,train_size=0.8,shuffle=True)

#test data split into x and y, maybe can remove this function
def test_data(df_tst,target_response):
    x=df_tst.drop(target_response,axis=1)
    y=df_tst[target_response]
    return x,y

#visualize the training lazzy classification result
def viz_clf_result(df):
    fig = go.Figure(
    go.Bar(
    x=df.index,
    y=df['Time Taken'],
    name="Training Time"
    )

    )
    for i in df.columns.drop('Time Taken'):
        fig.add_trace(
            go.Line(
            x=df.index,
            y=df[i],
            yaxis="y2",
            name=i
            )
        )
    fig.update_layout(
        legend=dict(
            orientation="h",
            entrywidth=500,
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
            ),
        yaxis=dict(
            title=dict(text='Training Time'),
            side='left',
        ),
        yaxis2=dict(
            title=dict(text='Model Performance'),
            side='right',
            #range=[0.86,0.93],
            overlaying='y',
            tickmode='sync',
        )
    )
    fig.show()
    
#rank the model performance by score their performance positions. At first time I wrongly put time spent rank sorb descending, and XGB Boost was the best. I have correct time spent rank as ascending but I still contine the practise using XGBoost
def clf_rank(df):
    data=pd.DataFrame()
    rank_df=pd.DataFrame()
    data=df.copy()
    for i in df.columns:
        if i!='Time Taken':
            rank_df[f'{i}_rank']=data[i].rank(axis=0,method='average',ascending=False)

        if i=='Time Taken':
            rank_df[f'{i}_rank']=data[i].rank(axis=0,method='average',ascending=True)

    rank_df['Avg_rank_score']=rank_df.mean(axis=1)
    result=pd.concat([df,rank_df],axis=1)
    result=result.sort_values(by='Avg_rank_score')
    return result

#This is objecctive function for hyperopt. I have added 5 cross validations (CV) such that each time the objective function will test hyperopt with 5 different splits of training data, and return the lowest score to hyperopt
#Score is -1*f1*precision
def objective_cv(space):
    hp_clf=xgb.XGBClassifier(
        max_depth = int(space['max_depth']),
        gamma = space['gamma'],
        reg_alpha = int(space['reg_alpha']),
        reg_lambda = int(space['reg_lambda']),
        #min_child_weight=int(space['min_child_weight']),
        #colsample_bytree=int(space['colsample_bytree']),
        eta=space['eta'],
        sub_sample=space['subsample'],
        objective=space['objective'],
        importance_type=space['importance_type'],
        tree_method=space['tree_method'],
        #tree_method='hist',
        #device='cuda',
        seed=int(space['seed']),
        booster=space['booster'],
        n_estimators =1000,
        n_thread=-1,
    )
    #rotate and select the val and train data from the 5 folds, and return the min score of the 5 folds to the hperopt fmin() optimization function
    a=[k1,k2,k3,k4,k5]
    cv_score=[100,100,100,100,100]
    for i in range(len(a)):
        b=a.copy()
        #assign validation set
        x_val1=b[i].drop('y',axis=1)
        y_val1=b[i]['y']
        #assing training set
        del b[i]
        tr=pd.DataFrame()
        for j in range(len(b)):
            tr=pd.concat([tr,b[j]],axis=0)
        x_tr1=tr.drop('y',axis=1)
        y_tr1=tr['y']

        evaluation =[(x_val1,y_val1)]

        hp_clf.fit(x_tr1,
                   y_tr1,
                   eval_set=evaluation,
                   early_stopping_rounds=10,verbose=False)
    
        pred_ = hp_clf.predict(x_val1)
        f1_= f1_score(y_val1, pred_>0.5)
        precision_= precision_score(y_val1, pred_>0.5)
        accuracy_=accuracy_score(y_val1, pred_>0.85)
        #print(f"CV{i+1} F1:{round(f1,4)} Precision:{round(precision,4)} Accuracy:{round(accuracy,4)} preferred_score{round(precision,4)}*{round(f1,4)}={round(precision*f1,4)}")
        cv_score[i]=f1_*precision_
    best_score=min(cv_score)
    #print(f"This literation best CV score={round(best_score,4)}")
        

    return {'loss': -best_score, 'status': STATUS_OK }

#hyperopt function with 300 eval
def hyperopt_hp_cv(objective, xgb_space):
    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = xgb_space,
                            algo = tpe.suggest,
                            max_evals = 300,
                            trials = trials,
                            )
    return best_hyperparams

#Fit model with parameter dict and space
def opt_xgb_model(xgb_space,hpdict):

    xgb_model=XGBClassifier(
        #colsample_bytree=hpdict['colsample_bytree'],
        eta=hpdict['eta'],
        gamma=hpdict['gamma'],
        max_depth=int(hpdict['max_depth']),
        #min_child_weight=int(hpdict['min_child_weight']),
        reg_alpha=hpdict['reg_alpha'],
        reg_lambda=hpdict['reg_lambda'],
        subsample=hpdict['subsample'],     
        objective=hp_objective[hpdict['objective']],
        importance_type=hp_importance_type[hpdict['importance_type']],
        tree_method=hp_tree_method[hpdict['tree_method']],
        seed=int(hpdict['seed']),
        booster=hp_booster[hpdict['booster']],
        nthread=-1,
        scale_pos_weight=1,
        n_estimators=1000,
        )

    xgb_model.fit(
        xgb_space['x_tr'],
        xgb_space['y_tr'],
        early_stopping_rounds=10,
        eval_set=[(xgb_space['x_val'],xgb_space['y_val'])],
        verbose=False
        )

    return xgb_model

#plot model result
def xgb_result(xgb_model,x,y_true):
    plt.figure(figsize=(9,4))
    y_pred=xgb_model.predict(x)
    cfm=confusion_matrix(y_true,y_pred)
    sns.heatmap(cfm,annot=True, fmt='d', cmap='Blues')
    plt.title("confusion matrix of XGB")
    plt.xlabel("Predict")
    plt.ylabel("Actual");
    print(classification_report(y_true, y_pred,target_names=['Not Subscribe 0','Subscribe 1']))

#similar objective function, but with score=precision
def objective_precision(space):
    hp_clf=xgb.XGBClassifier(
        max_depth = int(space['max_depth']),
        gamma = space['gamma'],
        reg_alpha = int(space['reg_alpha']),
        reg_lambda = int(space['reg_lambda']),
        #min_child_weight=int(space['min_child_weight']),
        #colsample_bytree=int(space['colsample_bytree']),
        eta=space['eta'],
        sub_sample=space['subsample'],
        objective=space['objective'],
        importance_type=space['importance_type'],
        tree_method=space['tree_method'],
        #tree_method='hist',
        #device='cuda',
        seed=int(space['seed']),
        booster=space['booster'],
        n_estimators =1000,
        n_thread=-1,
    )
    
    df1,k1=train_test_split(space['pro_df_tr'],train_size=0.8,shuffle=True)
    df1,k2=train_test_split(df1,train_size=(0.75),shuffle=True)
    df1,k3=train_test_split(df1,train_size=(2/3),shuffle=True)
    k5,k4=train_test_split(df1,train_size=0.5,shuffle=True)

    a=[k1,k2,k3,k4,k5]
    cv_score=[100,100,100,100,100]
    for i in range(len(a)):
        b=a.copy()
        #assign validation set
        x_val1=b[i].drop('y',axis=1)
        y_val1=b[i]['y']
        #assing training set
        del b[i]
        tr=pd.DataFrame()
        for j in range(len(b)):
            tr=pd.concat([tr,b[j]],axis=0)
        x_tr1=tr.drop('y',axis=1)
        y_tr1=tr['y']

        evaluation =[(x_val1,y_val1)]

        hp_clf.fit(x_tr1,
                   y_tr1,
                   eval_set=evaluation,
                   early_stopping_rounds=10,verbose=False)
    
        pred_ = hp_clf.predict(x_val1)
        f1_= f1_score(y_val1, pred_>0.5)
        precision_= precision_score(y_val1, pred_>0.5)
        accuracy_=accuracy_score(y_val1, pred_>0.85)
        #print(f"CV{i+1} F1:{round(f1,4)} Precision:{round(precision,4)} Accuracy:{round(accuracy,4)} preferred_score{round(precision,4)}*{round(f1,4)}={round(precision*f1,4)}")
        cv_score[i]=precision_
    best_score=min(cv_score)
    #print(f"This literation best CV score={round(best_score,4)}")
        

    return {'loss': -best_score, 'status': STATUS_OK }

#similar objective function, but with score=f1
def objective_f1(space):
    hp_clf=xgb.XGBClassifier(
        max_depth = int(space['max_depth']),
        gamma = space['gamma'],
        reg_alpha = int(space['reg_alpha']),
        reg_lambda = int(space['reg_lambda']),
        #min_child_weight=int(space['min_child_weight']),
        #colsample_bytree=int(space['colsample_bytree']),
        eta=space['eta'],
        sub_sample=space['subsample'],
        objective=space['objective'],
        importance_type=space['importance_type'],
        tree_method=space['tree_method'],
        #tree_method='hist',
        #device='cuda',
        seed=int(space['seed']),
        booster=space['booster'],
        n_estimators =1000,
        n_thread=-1,
    )
    
    df1,k1=train_test_split(space['pro_df_tr'],train_size=0.8,shuffle=True)
    df1,k2=train_test_split(df1,train_size=(0.75),shuffle=True)
    df1,k3=train_test_split(df1,train_size=(2/3),shuffle=True)
    k5,k4=train_test_split(df1,train_size=0.5,shuffle=True)

    a=[k1,k2,k3,k4,k5]
    cv_score=[100,100,100,100,100]
    for i in range(len(a)):
        b=a.copy()
        #assign validation set
        x_val1=b[i].drop('y',axis=1)
        y_val1=b[i]['y']
        #assing training set
        del b[i]
        tr=pd.DataFrame()
        for j in range(len(b)):
            tr=pd.concat([tr,b[j]],axis=0)
        x_tr1=tr.drop('y',axis=1)
        y_tr1=tr['y']

        evaluation =[(x_val1,y_val1)]

        hp_clf.fit(x_tr1,
                   y_tr1,
                   eval_set=evaluation,
                   early_stopping_rounds=10,verbose=False)
    
        pred_ = hp_clf.predict(x_val1)
        f1_= f1_score(y_val1, pred_>0.5)
        precision_= precision_score(y_val1, pred_>0.5)
        accuracy_=accuracy_score(y_val1, pred_>0.85)
        #print(f"CV{i+1} F1:{round(f1,4)} Precision:{round(precision,4)} Accuracy:{round(accuracy,4)} preferred_score{round(precision,4)}*{round(f1,4)}={round(precision*f1,4)}")
        cv_score[i]=f1_
    best_score=min(cv_score)
    #print(f"This literation best CV score={round(best_score,4)}")
        

    return {'loss': -best_score, 'status': STATUS_OK }

#fine tune with narrower search space 
def xgb_hp_opt2(objective,xgb_space,xgb_hpdict):

    #XGB cat type hyperparameters setup
    hp_objective=['binary:logistic','binary:logitraw','binary:hinge']
    hp_importance_type=['weight','gain','cover','total_gain','total_cover']
    hp_tree_method=['exact','approx','hist']
    hp_booster=['gbtree','gblinear','dart']
    #XGB hyper other parameters setup
    xgb_space_v2={'max_depth': hp.quniform("max_depth",max(0,xgb_hpdict["max_depth"]-4),xgb_hpdict["max_depth"]+4 ,1),
                'gamma': hp.uniform ('gamma', max(0,xgb_hpdict["gamma"]-6),xgb_hpdict["gamma"]+6),
                'reg_alpha' : hp.quniform('reg_alpha', max(0,xgb_hpdict["reg_alpha"]-2),xgb_hpdict["reg_alpha"]+2,1),
                'reg_lambda' : hp.uniform('reg_lambda', max(0,xgb_hpdict["reg_lambda"]-800),xgb_hpdict["reg_lambda"]+800),
                #'colsample_bytree' : hp.quniform('colsample_bytree',max(0,xgb_hpdict["colsample_bytree"]-0.1),min(xgb_hpdict["colsample_bytree"]+0.1,1),0.05),
                #'min_child_weight' : hp.quniform('min_child_weight', max(0,xgb_hpdict["min_child_weight"]-10),xgb_hpdict["min_child_weight"]+10, 1),
                'eta':hp.uniform ('eta', max(0,xgb_hpdict['eta']-0.2),min(xgb_hpdict['eta']+0.2,1)),
                'subsample':hp.uniform ('subsample', max(0,xgb_hpdict['subsample']-0.2),min(xgb_hpdict['subsample']+0.2,1)),
                'objective':hp.choice('objective',hp_objective),
                'importance_type':hp.choice('importance_type',hp_importance_type),
                'tree_method':hp.choice('tree_method',hp_tree_method),
                'seed':hp.quniform("seed", max(0,xgb_hpdict['seed']-2000),xgb_hpdict['seed']+2000,1),
                'booster':hp.choice('booster',hp_booster),
                'x_tr':x_tr,
                'x_val':x_val,
                'y_tr':y_tr,
                'y_val':y_val,
                'pro_df_tr':pro_df_tr,
                }
    hpdict=hyperopt_hp_cv(objective, xgb_space_v2)

    return hpdict

#similar but with narrow range
def xgb_hp_opt3(objective,xgb_space,xgb_hpdict):

    #XGB cat type hyperparameters setup
    hp_objective=['binary:logistic','binary:logitraw','binary:hinge']
    hp_importance_type=['weight','gain','cover','total_gain','total_cover']
    hp_tree_method=['exact','approx','hist']
    hp_booster=['gbtree','gblinear','dart']
    #XGB hyper other parameters setup
    xgb_space_v2={'max_depth': hp.quniform("max_depth",max(0,xgb_hpdict["max_depth"]-1),xgb_hpdict["max_depth"]+1 ,1),
                'gamma': hp.uniform ('gamma', max(0,xgb_hpdict["gamma"]-1.5),xgb_hpdict["gamma"]+1.5),
                'reg_alpha' : hp.quniform('reg_alpha', max(0,xgb_hpdict["reg_alpha"]-0.5),xgb_hpdict["reg_alpha"]+0.5,1),
                'reg_lambda' : hp.uniform('reg_lambda', max(0,xgb_hpdict["reg_lambda"]-200),xgb_hpdict["reg_lambda"]+200),
                #'colsample_bytree' : hp.quniform('colsample_bytree',max(0,xgb_hpdict["colsample_bytree"]-0.1),min(xgb_hpdict["colsample_bytree"]+0.1,1),0.05),
                #'min_child_weight' : hp.quniform('min_child_weight', max(0,xgb_hpdict["min_child_weight"]-10),xgb_hpdict["min_child_weight"]+10, 1),
                'eta':hp.uniform ('eta', max(0,xgb_hpdict['eta']-0.05),min(xgb_hpdict['eta']+0.05,1)),
                'subsample':hp.uniform ('subsample', max(0,xgb_hpdict['subsample']-0.05),min(xgb_hpdict['subsample']+0.05,1)),
                'objective':hp.choice('objective',hp_objective),
                'importance_type':hp.choice('importance_type',hp_importance_type),
                'tree_method':hp.choice('tree_method',hp_tree_method),
                'seed':hp.quniform("seed", max(0,xgb_hpdict['seed']-500),xgb_hpdict['seed']+500,1),
                'booster':hp.choice('booster',hp_booster),
                'x_tr':x_tr,
                'x_val':x_val,
                'y_tr':y_tr,
                'y_val':y_val,
                'pro_df_tr':pro_df_tr,
                }
    hpdict=hyperopt_hp_cv(objective, xgb_space_v2)

    return hpdict

#a looping function to run xg_hp_opt few time ('loop'). The hyperparametr is run first time in standard way, then feed into this function to move the search boundary each time with narrow range
def hp_opt_loop2(objective_cv,objective_precision,objective_f1,xgb_space,xgb_hpdict,loop):
    loop_hdict=[1,2,3]
    for i in range(loop):
        #print('Loop',i+1,'Precision opt:')
        #xgb_hpdict=xgb_hp_opt2(objective_precision,xgb_space,xgb_hpdict)
        #print('Loop',i+1,'F1 opt:')
        #xgb_hpdict=xgb_hp_opt2(objective_f1,xgb_space,xgb_hpdict)
        print('Loop',i+1,'CV opt')
        xgb_hpdict=xgb_hp_opt2(objective_cv,xgb_space,xgb_hpdict)
        loop_hdict[i]=xgb_hpdict

    return loop_hdict

#similar but use xgb_hp_opt3
def hp_opt_loop3(objective_cv,objective_precision,objective_f1,xgb_space,xgb_hpdict,loop):
    loop_hdict=[1,2,3]
    for i in range(loop):
        #print('Loop',i+1,'Precision opt:')
        #xgb_hpdict=xgb_hp_opt2(objective_precision,xgb_space,xgb_hpdict)
        #print('Loop',i+1,'F1 opt:')
        #xgb_hpdict=xgb_hp_opt2(objective_f1,xgb_space,xgb_hpdict)
        print('Loop',i+1,'CV opt')
        xgb_hpdict=xgb_hp_opt3(objective_cv,xgb_space,xgb_hpdict)
        loop_hdict[i]=xgb_hpdict

    return loop_hdict

def global_shap_importance(model, X):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model 
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance

def feat_rank(xgb_model):
        feat_gain=xgb_model.get_booster().get_score(importance_type='gain')
        feat_gain_df=pd.DataFrame(data=feat_gain.values(),index=feat_gain.keys(),columns=['gain'])
        feat_weight=xgb_model.get_booster().get_score(importance_type='weight')
        feat_weight_df=pd.DataFrame(data=feat_weight.values(),index=feat_weight.keys(),columns=['weight'])
        feat_cover=xgb_model.get_booster().get_score(importance_type='cover')
        feat_cover_df=pd.DataFrame(data=feat_cover.values(),index=feat_cover.keys(),columns=['cover'])
        feat_df=pd.concat([feat_gain_df,feat_weight_df,feat_cover_df],axis=1)

        data=pd.DataFrame()
        rank_df=pd.DataFrame()
        data=feat_df.copy()
        for i in feat_df.columns:
                rank_df[f'{i}_rank']=data[i].rank(axis=0,method='average',ascending=False)

        rank_df['Avg_rank_score']=rank_df.mean(axis=1)
        result=pd.concat([feat_df,rank_df],axis=1)
        result= result.sort_values(by='Avg_rank_score')
        return result