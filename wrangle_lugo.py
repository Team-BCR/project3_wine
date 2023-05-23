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
    df = pd.DataFrame()
    for file in files:
        data = pd.read_csv(file)
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