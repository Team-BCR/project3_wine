import pandas as pd
import numpy as np
from env import get_db_url
import os

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
def check_file_exists(fn, query, url):
    """
    This function will:
    - check if file exists in my local directory, if not, pull from sql db
    - read the given `query`
    - return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df
    
# ----------------------------------------------------------------------------------
def get_wine_data():                          
    # How to import a database from Data.world
    files = ['winequality-red.csv', 'winequality-white.csv']
    wine_type = ['red_wine', 'white_wine']
    
    df = pd.DataFrame()
    for i, file in enumerate(files):
        data = pd.read_csv(file)
        data['wine_type'] = wine_type[i]
        df = pd.concat([df, data], axis=0)
    df.to_csv('merged_winequality.csv', index=False)

#     files = 'zillow.csv'
#     df = check_file_exists(filename, query, url)
    
#     # Drop duplicate rows in column: 'parcelid', keeping max transaction date
#     df = df.drop_duplicates(subset=['parcelid'])
    
#     # rename columns
#     df.columns
#     df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms',
#                             'calculatedfinishedsquarefeet':'area','taxvaluedollarcnt':'property_value',
#                             'fips':'county','transaction_0':'transaction_year',
#                             'transaction_1':'transaction_month','transaction_2':'transaction_day'})
    
#     # total outliers removed are 6029 out of 52442
#     # # Look at properties less than 1.5 and over 5.5 bedrooms (Outliers were removed)
#     # df = df[~(df['bedrooms'] < 1.5) & ~(df['bedrooms'] > 5.5)]

#     # Look at properties less than .5 and over 4.5 bathrooms (Outliers were removed)
#     df = df[~(df['bathrooms'] < .5) & ~(df['bathrooms'] > 4.5)]

#     # Look at properties less than 1906.5 and over 2022.5 years (Outliers were removed)
#     df = df[~(df['yearbuilt'] < 1906.5) & ~(df['yearbuilt'] > 2022.5)]

#     # Look at properties less than -289.0 and over 3863.0 area (Outliers were removed)
#     df = df[~(df['area'] < -289.0) & ~(df['area'] > 3863.0)]

#     # Look at properties less than -444576.5 and over 1257627.5 property value (Outliers were removed)
#     df = df[~(df['property_value'] < -444576.5) &  ~(df['property_value'] > 1257627.5)]
    
#     # replace missing values with "0"
#     df = df.fillna({'bedrooms':0,'bathrooms':0,'area':0,'property_value':0,'county':0})
    
#     # drop any nulls in the dataset
#     df = df.dropna()
    
#     # drop all duplicates
#     df = df.drop_duplicates(subset=['parcelid'])
    
#     # change the dtype from float to int  
#     df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']] = df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']].astype(int)
    
#     # rename the county codes inside county
#     df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
#     # get dummies and concat to the dataframe
#     dummy_tips = pd.get_dummies(df[['county']], dummy_na=False, drop_first=[True, True])
#     df = pd.concat([df, dummy_tips], axis=1)
    
#     # dropping these columns for right now until I find a use for them
#     df = df.drop(columns =['parcelid','transactiondate','transaction_year','transaction_month','transaction_day'])
    
#     # Define the desired column order
#     new_column_order = ['bedrooms','bathrooms','area','yearbuilt','county','county_Orange','county_Ventura','property_value',]

#     # Reindex the DataFrame with the new column order
#     df = df.reindex(columns=new_column_order)

#     # write the results to a CSV file
#     df.to_csv('df_prep.csv', index=False)

#     # read the CSV file into a Pandas dataframe
#     prep_df = pd.read_csv('df_prep.csv')
    
    return df
# ----------------------------------------------------------------------------------
def prep_wine_data(df):
    
    new_col_name = []

    for col in df.columns:
        new_col_name.append(col.lower().replace(' ', '_'))

    df.columns = new_col_name
    
    return df

# ----------------------------------------------------------------------------------
def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing


# ----------------------------------------------------------------------------------
# def nulls_by_row2(df, index_id = 'customer_id'):
#     """
#     """
#     num_missing = df.isnull().sum(axis=1)
#     pct_miss = (num_missing / df.shape[1]) * 100
#     row_missing = df.isnull().sum()
    
#     rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': pct_miss, 'num_rows':row_missing})

#     rows_missing = df.merge(rows_missing,
#                         left_index=True,
#                         right_index=True)[['num_cols_missing', 'percent_cols_missing','num_rows']].drop('index', axis=1)
    
#     return rows_missing #.sort_values(by='num_cols_missing', ascending=False)

def nulls_by_row(df, index_id='customer_id'):
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    row_missing = num_missing.value_counts().sort_index()

    rows_missing = pd.DataFrame({
        'num_cols_missing': num_missing,
        'percent_cols_missing': pct_miss,
        'num_rows': row_missing
    }).reset_index()

    result_df = df.merge(rows_missing, left_index=True, right_on='index').drop('index', axis=1)[['num_cols_missing', 'percent_cols_missing', 'num_rows']]

    return result_df #[['num_cols_missing', 'percent_cols_missing', 'num_rows']]
# ----------------------------------------------------------------------------------
def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

# ----------------------------------------------------------------------------------
def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols
# ----------------------------------------------------------------------------------
def remove_columns(df, cols_to_remove):
    """
    This function will:
    - take in a df and list of columns (you need to create a list of columns that you would like to drop under the name 'cols_to_remove')
    - drop the listed columns
    - return the new df
    """
    df = df.drop(columns=cols_to_remove)
    
    return df

# ----------------------------------------------------------------------------------
def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df


# ----------------------------------------------------------------------------------
def data_prep(df, col_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - list of columns
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - removes unwanted columns
    - remove rows and columns that contain a high proportion of missing values
    - returns cleaned df
    """
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df
# ----------------------------------------------------------------------------------
def get_upper_outliers(s, m=1.5):
    '''
    Given a series and a cutoff value, m, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + (m * iqr)
    
    return s.apply(lambda x: max([x - upper_bound, 0]))

# ----------------------------------------------------------------------------------
def add_upper_outlier_columns(df, m=1.5):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('number'):
        df[col + '_outliers_upper'] = get_upper_outliers(df[col], m)
    return df

# ----------------------------------------------------------------------------------
# remove all outliers put each feature one at a time
def outlier(df, feature, m=1.5):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound