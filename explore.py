# IMPORTS
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

# FUNCTIONS
# defining a function to plot alcohol vs quality
def get_alcohol_qual_plot(df, x='alcohol', y='quality'):
    """
    This function will
    - accept a dataframe, and two strings (default 'alchol' and 'quality')
    - make a regplot of alcohol vs quality from the dataframe
    """
    sns.regplot(data = df, x=x, y=y, line_kws = {'color': 'red'})
    plt.xlabel('alcohol % by volume')
    plt.ylabel('wine quality score')
    plt.title('Is alcohol associated with quality?')
    plt.show()
    
# defining a function to plot chloride levels vs quality
def get_chloride_qual_plot(df, x='chlorides', y='quality'):
    """
    This function will
    - accept a dataframe, and two strings (default 'chlorides' and 'quality')
    - make a regplot of chlorides vs quality from the dataframe
    """
    sns.regplot(data=df, x='chlorides', y='quality', line_kws = {'color': 'red'})
    plt.xlabel('chlorides (g/L)')
    plt.ylabel('wine quality score')
    plt.title('Are chloride levels associated with quality?')
    plt.show()
    
# defining a function to plot residual_sugar vs quality
def get_res_sugar_quality_plot(df, x='residual_sugar', y='quality'):
    """
    This function will
    - accept a dataframe, and two strings (default 'residual_sugar' and 'quality')
    - make a regplot of residual_sugar vs quality from the dataframe
    """
    sns.regplot(data=df, x='residual_sugar', y='quality', line_kws = {'color': 'red'})
    plt.xlabel('residual sugar (g/L)')
    plt.ylabel('wine quality score')
    plt.title('Is residual_sugar associated with qualirty?')
    plt.show()
    
# defining a function to plot residual_sugar vs quality
def get_alcohol_density_plot(df, x='alcohol', y='density'):
    """
    This function will
    - accept a dataframe, and two strings (default 'alcohol' and 'density')
    - make a regplot of alcohol vs density from the dataframe
    """
    sns.regplot(data=df, x='alcohol', y='density', line_kws = {'color': 'red'})
    plt.xlabel('alcohol % by volume')
    plt.ylabel('density (g/l)')
    plt.title('How strongly are alcohol and density associated?')
    plt.show()