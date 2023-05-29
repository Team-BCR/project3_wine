### Imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pydataset import data
from scipy import stats as stat

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import lugo_wrangle as wl
import wrangle as wra
import corey_model as cm


### cluster, stats functions

def density_chlorides_cluster(df):
    '''This function takes in wine data named df, splits it into train, validation, and test sets, scales the data, and performs K-means clustering on the 'density' and 'chlorides' columns. It then assigns the cluster number to a new column in the dataframe. Finally, it bins the 'quality' variable into two categories and returns the resulting training data.'''
    
    # drop the 'wine_type' column
    # df = df.drop('wine_type', axis=1)
    
    # split the data
    tr, val, ts = wra.get_split(df)
    
    # specify the target variable
    target = 'quality'
    
    # get the X and y variables to scale and baseline
    X_tr, X_val, X_ts, y_tr, y_val, y_ts, to_scale, baseline = wra.get_Xs_ys_to_scale_baseline(tr, val, ts, target)
    
    # scale the data
    X_tr_sc, X_val_sc, X_ts_sc = wra.scale_data(X_tr, X_val, X_ts, to_scale)
    
    # extract the 'density' and 'chlorides' columns
    X_tr2 = X_tr_sc[['density', 'chlorides']]
    
    # perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
    kmeans.fit(X_tr2)
    X_tr2["cluster"] = kmeans.predict(X_tr2)
    
    # bin the 'quality' variable
    tr['quality_cat'] = pd.cut(tr['quality'], bins=[2, 6, 9], labels=['3-6', '7-9'])
    y_tr2 = tr[['quality_cat']]
    
    return tr, val, ts, X_tr, X_val, X_ts, y_tr, y_val, y_ts, to_scale, baseline, X_tr_sc, X_val_sc, X_ts_sc, X_tr2, y_tr2


def density_chlorides_cluster_plot(X_tr2, y_tr2):
    '''This function takes in the training data X_tr2 and target variable y_tr2, and creates two scatter plots using Seaborn's relplot function. The first plot shows the clusters, and the second plot shows the clusters with the 'quality_cat' variable added as the hue. The function then displays the plots.'''
    
    # create the first scatter plot
    sns.relplot(data=X_tr2,
                x='density',
                y='chlorides',
                hue='cluster',
                col='cluster',
                col_wrap=3)

    # create the second scatter plot
    sns.relplot(data=X_tr2.join(y_tr2),
                x='density',
                y='chlorides',
                hue='quality_cat',
                col='cluster',
                col_wrap=3)

    # show the plots
    plt.show()
        
def density_chlorides_cluster_ttest(X_tr2, y_tr):
    '''This function takes in the training data X_tr2 and target variable y_tr, extracts the rows labeled '0' and '2' in the 'cluster' column and the second column of the dataframe, and performs a two-tailed t-test between the two samples. The function then prints the t-statistic and p-value multiplied by 2.'''
    
    temp_df = pd.concat([X_tr2, y_tr], axis=1)
    cluster_0 = temp_df[temp_df['cluster'] == 0][temp_df.columns[1]].astype(float)
    cluster_2 = temp_df[temp_df['cluster'] == 2][temp_df.columns[1]].astype(float)
    t_stat, p_value = stat.ttest_ind(cluster_0, cluster_2, equal_var=False)
    print("t-statistic:", t_stat)
    print("p-value:", p_value * 2) # multiply p-value by 2 for a two-tailed test

    
    
def get_metrics_with_cluster(X_tr2, X_tr_sc, y_tr, X_val_sc, y_val, alpha=1, power=2, degrees=2):
    """
    This function will
    - accept X_tr_sc, y_tr, X_val_sc, y_val
    - accept values for alpha, power, and degrees; default values are 1/2/2
        - alpha is a hyperparameter for LassoLARS
        - power is a hyperparameter for Polynomial Regressioin
        - degrees is a hyperparameter for GLM
    - add cluster column to X_tr_sc and X_val_sc
    - create dummy variables for cluster column with drop_first=True
    - drop cluster column from X_tr_sc and X_val_sc
    - drop rows with missing values from X_val_sc
    - call get_reg_model_metrics_df function and return the resulting dataframe
    """
    # add cluster column to X_tr_sc and X_val_sc
    X_tr_sc = pd.concat([X_tr_sc, X_tr2['cluster']], axis=1)
    X_val_sc = pd.concat([X_val_sc, X_tr2['cluster']], axis=1)
    
    # create dummy variables for cluster column with drop_first=True
    dummies = pd.get_dummies(X_tr_sc['cluster'], prefix='cluster', drop_first=True)
    X_tr_sc = pd.concat([X_tr_sc, dummies], axis=1)
    dummies = pd.get_dummies(X_val_sc['cluster'], prefix='cluster', drop_first=True)
    X_val_sc = pd.concat([X_val_sc, dummies], axis=1)
    
    # drop cluster column from X_tr_sc and X_val_sc
    X_tr_sc = X_tr_sc.drop('cluster', axis=1)
    X_val_sc = X_val_sc.drop('cluster', axis=1)
    
    # drop rows with missing values from X_val_sc
    X_val_sc = X_val_sc.dropna()
    
    # call get_reg_model_metrics_df function and return the resulting dataframe
    return cm.get_reg_model_metrics_df(X_tr_sc, y_tr, X_val_sc, y_val, alpha=alpha, power=power, degrees=degrees)

def plot_cluster_2(X_tr, y_tr):
    tr = X_tr.join(y_tr)
    tr['quality_cat'] = pd.cut(tr['quality'], bins=[2, 6, 9], labels=['3-6', '7-9'])

    y_tr2 = tr[['quality_cat']]

    X_tr3 = X_tr[['alcohol', 'residual_sugar']]

    #make it
    kmeans = KMeans(n_clusters = 3, random_state=123, n_init=10)

    #fit it
    kmeans.fit(X_tr3)

    #use it
    kmeans.predict(X_tr3)

    # And assign the cluster number to a column on the dataframe
    X_tr3["cluster"] = kmeans.predict(X_tr3)

    # create the first scatter plot
    sns.relplot(data=X_tr3,
               x='alcohol',
               y='residual_sugar',
               hue='cluster',
               col='cluster',
               col_wrap=3)

    # create the second scatter plot
    sns.relplot(data=X_tr3.join(y_tr2),
               x='alcohol',
               y='residual_sugar',
               hue='quality_cat',
               col='cluster',
               col_wrap=3)

    # show the plots
    plt.show()
    

def plot_cluster_3(X_tr, y_tr):
    tr = X_tr.join(y_tr)
    tr['quality_cat'] = pd.cut(tr['quality'], bins=[2, 6, 9], labels=['3-6', '7-9'])

    y_tr2 = tr[['quality_cat']]
    
    X_tr4 = X_tr[['total_sulfur_dioxide', 'residual_sugar']]
    
    #make it
    kmeans = KMeans(n_clusters = 3, random_state=123, n_init=10)

    #fit it
    kmeans.fit(X_tr4)

    #use it
    kmeans.predict(X_tr4)
    
    # And assign the cluster number to a column on the dataframe
    X_tr4["cluster"] = kmeans.predict(X_tr4)
    
    # create the first scatter plot
    sns.relplot(data=X_tr4,
               x='total_sulfur_dioxide',
               y='residual_sugar',
               hue='cluster',
               col='cluster',
               col_wrap=3)

    # create the second scatter plot
    sns.relplot(data=X_tr4.join(y_tr2),
               x='total_sulfur_dioxide',
               y='residual_sugar',
               hue='quality_cat',
               col='cluster',
               col_wrap=3)

    # show the plots
    plt.show()
    