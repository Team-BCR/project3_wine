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

### cluster, stats functions

def density_chlorides_cluster(df):
    '''This function takes in wine data named df, splits it into train, validation, and test sets, scales the data, and performs K-means clustering on the 'density' and 'chlorides' columns. It then assigns the cluster number to a new column in the dataframe. Finally, it bins the 'quality' variable into two categories and returns the resulting training data.'''
    
    # drop the 'wine_type' column
    df = df.drop('wine_type', axis=1)
    
    # split the data
    tr, val, ts = wl.get_split(df)
    
    # specify the target variable
    target = 'quality'
    
    # get the X and y variables to scale and baseline
    X_tr, X_val, X_ts, y_tr, y_val, y_ts, to_scale, baseline = wl.get_Xs_ys_to_scale_baseline(tr, val, ts, target)
    
    # scale the data
    X_tr_sc, X_val_sc, X_ts_sc = wl.scale_data(X_tr, X_val, X_ts, to_scale)
    
    # extract the 'density' and 'chlorides' columns
    X_tr2 = X_tr_sc[['density', 'chlorides']]
    
    # perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
    kmeans.fit(X_tr2)
    X_tr2["cluster"] = kmeans.predict(X_tr2)
    
    # bin the 'quality' variable
    tr['quality_cat'] = pd.cut(tr['quality'], bins=[2, 6, 9], labels=['3-6', '7-9'])
    y_tr2 = tr[['quality_cat']]
    
    return X_tr2, y_tr2, y_tr

    
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
    

    
    
    
    