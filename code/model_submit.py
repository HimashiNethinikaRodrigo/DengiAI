import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

feature = pd.read_csv('../data/dengue_features_train.csv', infer_datetime_format=True)
label = pd.read_csv('../data/dengue_labels_train.csv')

df = pd.merge(feature, label, how='outer', on=label.columns.tolist()[:-1])

np.sum(df.isnull(), axis=0)
df = df.dropna(axis=0, thresh=20)
np.sum(df.isnull(), axis=0)
df.city.value_counts()

df_sj = df[df.city == 'sj'].copy()
df_iq = df[df.city == 'iq'].copy()
df.columns.tolist()

# Model
df = df.join(pd.get_dummies(df.city))
np.sum(df.isnull(), axis=0)
ignore_feature_list = ['city', 'ndvi_ne', 'week_start_date', 'total_cases']
predictors = [feature for feature in df.columns.tolist() if feature not in ignore_feature_list]
target = 'total_cases'
df_mean = df.fillna(df.mean())


def make_prediction(alg, train_df, test_df, predictors, target):
    alg.fit(train_df[predictors], train_df[target])
    predictions = alg.predict(test_df[predictors])

    result_df = test_df[['city', 'year', 'weekofyear']].copy()
    result_df['total_cases'] = predictions
    result_df.total_cases = result_df.total_cases.round()
    result_df.total_cases = result_df.total_cases.astype(int)

    return result_df


def processing_function(df):
    df.fillna(method='ffill', inplace=True)

    df_sj = df.loc[df.city == 'sj']
    df_iq = df.loc[df.city == 'iq']

    return df_sj, df_iq


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.hstack((ret[:n - 1], ret[n - 1:] / n))


lagging_feature_list = [ 'ndvi_nw',
                         'ndvi_se',
                         'ndvi_sw',
                         'precipitation_amt_mm',
                         'reanalysis_air_temp_k',
                         'reanalysis_avg_temp_k',
                         'reanalysis_dew_point_temp_k',
                         'reanalysis_max_air_temp_k',
                         'reanalysis_min_air_temp_k',
                         'reanalysis_precip_amt_kg_per_m2',
                         'reanalysis_relative_humidity_percent',
                         'reanalysis_sat_precip_amt_mm',
                         'reanalysis_specific_humidity_g_per_kg',
                         'reanalysis_tdtr_k',
                         'station_avg_temp_c',
                         'station_diur_temp_rng_c',
                         'station_max_temp_c',
                         'station_min_temp_c',
                         'station_precip_mm']


def add_lagging_feature(df, lagging_feature_list, n_lag=4, remove_nan_row=True):
    new_df = df.copy()

    for original_feature in lagging_feature_list:
        for n in range(n_lag):
            lagging_feature_name = original_feature + '_lag_' + str(n + 1)
            new_df.loc[:, lagging_feature_name] = new_df.loc[:, original_feature].shift(n + 1)

    new_df = new_df.iloc[n_lag:, :]

    for original_feature in lagging_feature_list:
        rolling_mean_feat_name = 'rolling_mean_' + original_feature
        rolling_std_feat_name = 'rolling_std_' + original_feature

        new_df.loc[:, rolling_mean_feat_name] = new_df[original_feature].rolling(window=3, center=False).mean()
        new_df.loc[:, rolling_std_feat_name] = new_df[original_feature].rolling(window=3, center=False).std()

    new_df.fillna(new_df.mean(), inplace=True)

    return new_df


sj_lag = df_sj.copy()
iq_lag = df_iq.copy()

sj_lag = add_lagging_feature(df_sj, lagging_feature_list)
iq_lag = add_lagging_feature(df_iq, lagging_feature_list)

rf_sj_lag = RandomForestRegressor(max_features=120, min_samples_split=70, n_estimators=840, max_depth=65,
                                  min_samples_leaf=3)

rf_iq_lag = RandomForestRegressor(max_features=10, min_samples_split=4, n_estimators=751, max_depth=76,
                                  min_samples_leaf=45)

# Check
test = pd.read_csv('../data/dengue_features_test.csv')
test = test.join(pd.get_dummies(test.city))

sj, iq = processing_function(df)
test_sj, test_iq = processing_function(test)

test_sj_lag = pd.concat([sj.iloc[-4:, sj.columns != 'total_cases'], test_sj])
test_iq_lag = pd.concat([iq.iloc[-4:, iq.columns != 'total_cases'], test_iq])

test_sj_lag = add_lagging_feature(test_sj_lag, lagging_feature_list)
test_iq_lag = add_lagging_feature(test_iq_lag, lagging_feature_list)

lagging_predictors = [feat for feat in sj_lag.columns.tolist()if feat not in ['city', 'sj', 'iq', 'total_cases', 'week_start_date'] ]

target = 'total_cases'

result_sj = make_prediction(rf_sj_lag, sj_lag, test_sj_lag, lagging_predictors, target)

result_iq =  make_prediction(rf_iq_lag, iq_lag, test_iq_lag, lagging_predictors, target)

result = pd.concat([result_sj, result_iq])

first_2_cases = result.iloc[:2,:]['total_cases'].values


result.total_cases = result.total_cases.rolling(window=3).mean()

result.iloc[ :2, -1] = first_2_cases

result.total_cases = result.total_cases.astype(int)

result.to_csv('../result/submission_1.csv', index=False)



