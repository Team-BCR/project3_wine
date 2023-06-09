import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

# Stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector

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
    
    return df
# ----------------------------------------------------------------------------------
def prep_wine_data(df):
    
    new_col_name = []

    for col in df.columns:
        new_col_name.append(col.lower().replace(' ', '_'))

    df.columns = new_col_name
    
    # drop any nulls in the dataset
    df = df.dropna()

    # get dummies and concat to the dataframe
    dummy_tips = pd.get_dummies(df[['wine_type']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_tips], axis=1)
           
    df = df.reset_index()
    df = df.drop(columns = ['index'])
    
    return df

# ----------------------------------------------------------------------------------
def get_split(df):
    '''
    train=tr
    validate=val
    test=ts
    test size = .2 and .25
    random state = 123
    '''  
    # split your dataset
    train_validate, ts = train_test_split(df, test_size=.2, random_state=123)
    tr, val = train_test_split(train_validate, test_size=.25, random_state=123)
    
    # printing the shape of each split
    print(f'Train: {tr.shape}')
    print(f'Validate: {val.shape}')
    print(f'Test: {ts.shape}')
    
    return tr, val, ts
# ----------------------------------------------------------------------------------
def get_Xs_ys_to_scale_baseline(tr, val, ts, target):
    '''
    tr = train
    val = validate
    ts = test
    target = target value
    '''

    # Separate the features (X) and target variable (y) for the training set
    X_tr, y_tr = tr.drop(columns=[target,]), tr[target]
    
    # Separate the features (X) and target variable (y) for the validation set
    X_val, y_val = val.drop(columns=[target,]), val[target]
    
    # Separate the features (X) and target variable (y) for the test set
    X_ts, y_ts = ts.drop(columns=[target,]), ts[target]
    
    # Get the list of columns to be scaled
    to_scale = X_tr.columns.tolist()
    
    # Calculate the baseline (mean) of the target variable in the training set
    baseline = y_tr.mean()
    
    # print baseline to make final_notebook cleaner
    print(f'Baseline Prediction (mean of quality) = {baseline}')
    
    # Return the separated features and target variables, columns to scale, and baseline
    return X_tr, X_val, X_ts, y_tr, y_val, y_ts, to_scale, baseline


# ----------------------------------------------------------------------------------
def scale_data(X,Xv,Xts,to_scale):
    '''
    X = X_train
    Xv = X_validate
    Xts = X_test
    to_scale, is found in the get_Xs_ys_to_scale_baseline
    '''
    
    #make copies for scaling
    X_tr_sc = X.copy()
    X_val_sc = Xv.copy()
    X_ts_sc = Xts.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(X[to_scale])

    #use the thing
    X_tr_sc[to_scale] = scaler.transform(X[to_scale])
    X_val_sc[to_scale] = scaler.transform(Xv[to_scale])
    X_ts_sc[to_scale] = scaler.transform(Xts[to_scale])
    
    return X_tr_sc, X_val_sc, X_ts_sc

# ----------------------------------------------------------------------------------
def metrics_reg(y, yhat):
    """
    y = y_train
    yhat = y_pred
    send in y_train, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


# ----------------------------------------------------------------------------------
def rfe(X,v,y,k=3):
    '''
    # X = X_train_scaled
    # v = X_validate_scaled
    # y = y_train
    # k = the number of features to select
    '''
    
    # make a model object to use in RFE process.
    # The model is here to give us metrics on feature importance and model score
    # allowing us to recursively reduce the number of features to reach our desired space
    model = LinearRegression()
    
    # MAKE the thing
    rfe = RFE(model, n_features_to_select=k)

    # FIT the thing
    rfe.fit(X, y)
    
    X_tr_rfe = pd.DataFrame(rfe.transform(X),index=X.index,
                                          columns = X.columns[rfe.support_])
    
    X_val_rfe = pd.DataFrame(rfe.transform(v),index=v.index,
                                      columns = v.columns[rfe.support_])
    
    top_k_rfe = X.columns[rfe.get_support()]
    
    return top_k_rfe, X_tr_rfe, X_val_rfe
# ----------------------------------------------------------------------------------
def get_models_dataframe(baseline, tr, y_tr, y_val, y_ts, X_tr_sc, X_val_sc, X_ts_sc):
    baseline_array = np.repeat(baseline, len(tr))
    rmse_tr, r2 = metrics_reg(y_tr, baseline_array)
    rmse_val = ''
    metrics_df = pd.DataFrame(data=[{'model': 'baseline','rmse train':rmse_tr ,'rmse validate': rmse_val, 'R2 validate': r2}])

    # OLS + RFE
    top_k_rfe, X_tr_rfe, X_val_rfe = rfe(X_tr_sc, X_val_sc, y_tr, 3) # use the 3 best features of RFE
    lr_rfe = LinearRegression()
    lr_rfe.fit(X_tr_rfe, y_tr)

    pred_lr_rfe_tr = lr_rfe.predict(X_tr_rfe)
    rmse_tr, r2_tr = metrics_reg(y_tr, pred_lr_rfe_tr)

    pred_lr_rfe_val = lr_rfe.predict(X_val_rfe)
    rmse_val, r2_val = metrics_reg(y_val, pred_lr_rfe_val)
    metrics_df.loc[1] = ['ols with RFE-top-3 features', rmse_tr, rmse_val, r2_val]
    
    # This print statement was to figure out which columns were top 3
    # might be nice to return in the future
    # print(X_tr_rfe.head())

    # OLS
    lr = LinearRegression()
    lr.fit(X_tr_sc, y_tr)

    pred_lr_tr = lr.predict(X_tr_sc)
    rmse_tr, r2 = metrics_reg(y_tr, pred_lr_tr)

    pred_lr_val = lr.predict(X_val_sc)
    rmse_val, r2 = metrics_reg(y_val, pred_lr_val)
    metrics_df.loc[2] = ['ols', rmse_tr, rmse_val, r2]

    # LARS
    lars = LassoLars(alpha=.5) # default is 1.  .5
    lars.fit(X_tr_sc, y_tr)

    pred_lars_tr = lars.predict(X_tr_sc)
    rmse_tr, r2 = metrics_reg(y_tr, pred_lars_tr)

    pred_lars_val = lars.predict(X_val_sc)
    rmse_val, r2 = metrics_reg(y_val, pred_lars_val)
    metrics_df.loc[3] = ['lars', rmse_tr, rmse_val, r2]

    # # Polynomial
    degrees = 2
    # for degree in degrees:
    pf = PolynomialFeatures(degree=degrees)
    X_tr_degree = pf.fit_transform(X_tr_sc)
    X_val_degree = pf.transform(X_val_sc)
    X_ts_degree = pf.transform(X_ts_sc)

    pr = LinearRegression()
    pr.fit(X_tr_degree, y_tr)

    pred_pr_tr = pr.predict(X_tr_degree)
    rmse_tr, r2 = metrics_reg(y_tr, pred_pr_tr)

    pred_pr_val = pr.predict(X_val_degree)
    rmse_val, r2 = metrics_reg(y_val, pred_pr_val)
    metrics_df.loc[4] = [f"poly_{degrees}D",rmse_tr, rmse_val, r2]


    # GLM
    # power = 0: Normal Distribution
    # power = 1: Poisson Distribution
    # power = 2: Gamma Distribution
    # power = 3: Inverse Gaussian Distribution
    glm = TweedieRegressor(power=0, alpha=.5) # default 1
    glm.fit(X_tr_sc, y_tr)

    pred_glm_tr = glm.predict(X_tr_sc)
    rmse_tr, r2 = metrics_reg(y_tr, pred_glm_tr)

    pred_glm_val = glm.predict(X_val_sc)
    rmse_val, r2 = metrics_reg(y_val, pred_glm_val)
    metrics_df.loc[5] = ['glm', rmse_tr, rmse_val, r2]

    return metrics_df, pred_lr_rfe_tr, pred_lr_tr, pred_lars_tr, pred_pr_tr, pred_glm_tr
# ----------------------------------------------------------------------------------
def test_best_model(X_ts_sc, y_ts, X_tr_sc, y_tr):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_tr_2d = pf.fit_transform(X_tr_sc)
    # transform & X_test_scaled
    X_ts_2d = pf.transform(X_ts_sc)

    #make it
    pr = LinearRegression()
    #fit it
    pr.fit(X_tr_2d, y_tr)

    #use it
    pred_ts = pr.predict(X_ts_2d)
    rmse, r2 = metrics_reg(y_ts, pred_ts)
    result = {'model Poly_2D': 'test', 'rmse': rmse, 'r2': r2}
    result = pd.DataFrame([result])
    return result

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
# remove all outliers put each feature one at a time
def outlier(df, feature, m=2):
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
# ----------------------------------------------------------------------------------

def execute_outlier(df):
    # total rows
    orig_shape = df.shape[0]
    
    # finding the lower and upper bound outliers for fixed acidity
    fix_acUP, fix_acLOW = outlier(df,'fixed_acidity')
    df = df[(df.fixed_acidity < fix_acUP) & (df.fixed_acidity > fix_acLOW)]
    col1 = df.shape[0]

    
    # finding the lower and upper bound outliers for volatile_acidity
    vol_acUP, vol_acLOW = outlier(df,'volatile_acidity')
    df = df[(df.volatile_acidity < vol_acUP) & (df.volatile_acidity > vol_acLOW)]
    col2 = df.shape[0]


    # finding the lower and upper bound outliers for citric_acid
    cit_acUP, cit_acLOW = outlier(df,'citric_acid')
    df = df[(df.citric_acid < cit_acUP) & (df.citric_acid > cit_acLOW)]
    col3 = df.shape[0]


    # finding the lower and upper bound outliers for residual_sugar
    res_sugUP, res_sugLOW = outlier(df,'residual_sugar')
    df = df[(df.residual_sugar < res_sugUP) & (df.residual_sugar > res_sugLOW)]
    col4 = df.shape[0]


    # finding the lower and upper bound outliers for chlorides
    chloUP, chloLOW = outlier(df,'chlorides')
    df = df[(df.chlorides < chloUP) & (df.chlorides > chloLOW)]
    col5 = df.shape[0]


    # finding the lower and upper bound outliers for free_sulfur_dioxide
    fsdUP, fsdLOW = outlier(df,'free_sulfur_dioxide')
    df = df[(df.free_sulfur_dioxide < fsdUP) & (df.free_sulfur_dioxide > fsdLOW)]
    col6 = df.shape[0]


    # finding the lower and upper bound outliers for total_sulfur_dioxide
    tsdUP, tsdLOW = outlier(df,'total_sulfur_dioxide')
    df = df[(df.total_sulfur_dioxide < tsdUP) & (df.total_sulfur_dioxide > tsdLOW)]
    col7 = df.shape[0]


    # finding the lower and upper bound outliers for density
    denUP, denLOW = outlier(df,'density')
    df = df[(df.density < denUP) & (df.density > denLOW)]
    col8 = df.shape[0]


    # finding the lower and upper bound outliers for ph
    phUP, phLOW = outlier(df,'ph')
    df = df[(df.ph < phUP) & (df.ph > phLOW)]
    col9 = df.shape[0]


    # finding the lower and upper bound outliers for sulphates
    sulUP, sulLOW = outlier(df,'sulphates')
    df = df[(df.sulphates < sulUP) & (df.sulphates > sulLOW)]
    col10 = df.shape[0]


    # finding the lower and upper bound outliers for alcohol
    alcUP, alcLOW = outlier(df,'alcohol')
    df = df[(df.alcohol < alcUP) & (df.alcohol > alcLOW)]
    col11 = df.shape[0]

    
    # print('Handaling OUTLIERS')
    # print(f"fixed_acidity: lower= {fix_acLOW}, upper= {fix_acUP}, new rows= {col1}\n")
    # print(f"volatile_acidity: lower= {vol_acLOW}, upper= {vol_acUP}, new rows= {col2}\n")
    # print(f"citric_acid: lower= {cit_acLOW}, upper= {cit_acUP}, new rows= {col3}\n")
    # print(f"residual_sugar: lower= {res_sugLOW}, upper= {res_sugUP}, new rows= {col4}\n")
    # print(f"chlorides: lower= {chloLOW}, upper= {chloUP}, new rows= {col5}\n")
    # print(f"free_sulfur_dioxide: lower= {fsdLOW}, upper= {fsdUP}, new rows= {col6}\n")    
    # print(f"total_sulfur_dioxide: lower= {tsdLOW}, upper= {tsdUP}, new rows= {col7}\n")    
    # print(f"density: lower= {denLOW}, upper= {denUP}, new rows= {col8}\n")    
    # print(f"ph: lower= {phLOW}, upper= {phUP}, new rows= {col9}\n")    
    # print(f"sulphates: lower= {sulLOW}, upper= {sulUP}, new rows= {col10}\n")    
    # print(f"alcohol: lower= {alcLOW}, upper= {alcUP}, new rows= {col11}\n")
    

    new_shape = df.shape[0]
    shape_rem = orig_shape-new_shape
    print(f"Total of rows originally: {orig_shape}")
    print(f"Total of rows removed: {shape_rem}")
    print(f"New total of rows: {new_shape}")
    
    return df

# ----------------------------------------------------------------------------------
def select_kbest(X,y,k):
    '''
    # X = X_train
    # y = y_train
    # k = the number of features to select (we are only sending two as of right now)
    '''
    
    # MAKE the thing
    kbest = SelectKBest(f_regression, k=k)

    # FIT the thing
    kbest.fit(X, y)
    
    # Create a DATAFRAME
    kbest_results = pd.DataFrame(
                dict(pvalues=kbest.pvalues_, feature_scores=kbest.scores_),
                index = X.columns)
    
    # we can apply this mask to the columns in our original dataframe
    top_k = X.columns[kbest.get_support()]
    
    return top_k

# ----------------------------------------------------------------------------------
def predict_vs_actual_graph(baseline, y_tr, pred_lr_tr, pred_pr_tr, pred_glm_tr):
    plt.scatter(pred_lr_tr, y_tr, label='linear regression')
    plt.scatter(pred_pr_tr, y_tr, label='polynominal 2deg')
    plt.scatter(pred_glm_tr, y_tr, label='glm')
    plt.plot(y_tr, y_tr, label='Perfect line of regression', color='grey')

    plt.axhline(baseline, ls=':', color='grey')
    plt.annotate("Baseline", (65, 81))

    plt.title("Where are predictions more extreme? More modest?")
    plt.ylabel("Actual Quality")
    plt.xlabel("Predicted Quality")
    plt.legend()

    plt.show()
    
# ----------------------------------------------------------------------------------
def residual_scatter(y_tr, pred_lr_tr, pred_pr_tr, pred_glm_tr):
    plt.axhline(label="No Error")

    plt.scatter(y_tr, pred_lr_tr - y_tr, alpha=.5, color="red", label="LinearRegression")
    plt.scatter(y_tr, pred_glm_tr - y_tr, alpha=.5, color="yellow", label="TweedieRegressor")
    plt.scatter(y_tr, pred_pr_tr - y_tr, alpha=.5, color="green", label="Polynomial 2deg ")

    plt.legend()
    plt.title("Do the size of errors change as the actual value changes?")
    plt.xlabel("Actual Quality")
    plt.ylabel("Residual: Predicted Quality - Actual Quality")

    plt.show()

# ----------------------------------------------------------------------------------
def distribution_actual_vs_predict(y_tr, pred_lr_tr, pred_pr_tr, pred_glm_tr):
    plt.hist(y_tr, color='blue', alpha=.5, label="Actual")
    plt.hist(pred_lr_tr, color='red', alpha=.5, label="LinearRegression")
    plt.hist(pred_glm_tr, color='yellow', alpha=.5, label="TweedieRegressor")
    plt.hist(pred_pr_tr, color='green', alpha=.5, label="Polynomial 2Deg")

    plt.xlabel("Quality")
    plt.ylabel("Number of Wines")
    plt.title("Comparing the Distribution of Actual to Predicted Quality")
    plt.legend()
    plt.show()
    
    
def wrangle_wine():
    """
    This function will
    - read in wine data from csv files into dataframe
    - prepare with prep_wine_data
    - remove outliers with execute_outliers
    - return df ready for splitting/exploring
    """
    df = get_wine_data()
    df = prep_wine_data(df)
    df = execute_outlier(df)

    # dropping the wine_type column which is a string and we have the data in an encoded column already
    obj_col = get_object_cols(df)
    df = df.drop(columns=obj_col)
    
    return df

def display_model_metrics(baseline,tr,y_tr,y_val,y_ts,X_tr_sc,X_val_sc,X_ts_sc):
    """
    This function will
    - accept a baseline value (float)
    - accept the train data set as well as the y splits and the X splits that are scaled
    - call get_models_dataframe
    - print a dataframe with the metrics results from get_models_dataframe (only on train and validate)
        -- (which begs the question why send in test; question for RL)-cah
    """
    
    metrics_df, pred_lr_rfe_tr, pred_lr_tr, pred_lars_tr, pred_pr_tr, pred_glm_tr = get_models_dataframe(baseline
                                                                                                         ,tr,y_tr,y_val,y_ts
                                                                                                         ,X_tr_sc,X_val_sc
                                                                                                         ,X_ts_sc)
    display(metrics_df)
    
    return