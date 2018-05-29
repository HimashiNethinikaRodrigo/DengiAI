from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

feature = pd.read_csv('./dengue_features_train.csv', infer_datetime_format=True)
label = pd.read_csv('./dengue_labels_train.csv')

df = pd.merge(feature, label, how='outer', on=label.columns.tolist()[:-1])

np.sum(df.isnull(), axis=0)
df = df.dropna(axis=0, thresh=20)
np.sum(df.isnull(), axis=0)
df.city.value_counts()

df_sj = df[df.city == 'sj'].copy()
df_iq = df[df.city == 'iq'].copy()
df.columns.tolist()


def first_model_fit(alg, dtrain, predictors, target, n_fold=10):
    alg.fit(dtrain[predictors], dtrain[target])
    dtrain_predictions = alg.predict(dtrain[predictors])

    cv_mae_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=n_fold,
                                   scoring='neg_mean_absolute_error')
    cv_mae_score = np.abs(cv_mae_score)

    # Print model report:
    print("\nModel Report: {0}-fold cross-validation".format(n_fold))
    print("Mean absolute error on training set: %.4g" % metrics.mean_absolute_error(dtrain[target].values,
                                                                                    dtrain_predictions))
    print("Mean - %.4g | Median - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_mae_score),
                                                                            np.median(cv_mae_score),
                                                                            np.std(cv_mae_score),
                                                                            np.min(cv_mae_score),
                                                                            np.max(cv_mae_score)))


def first_model(test_df, predictors, target, plotting=False, num_feature=10):
    base_result = test_df[target][:]
    base_result[:] = base_result.mean(axis=0)
    print("NAIVE MODEL: all predictions = mean value")
    print("Mean absolute error : %.4g" % np.sqrt(metrics.mean_squared_error(test_df[target].values, base_result)))
    print("\n", "----------------LINEAR REGRESSION------------------------")

    lm = LinearRegression(normalize=False)
    first_model_fit(lm, test_df, predictors, target)
    coef1 = pd.Series(lm.coef_, predictors).sort_values()
    if plotting:
        coef1.plot(kind='barh', title='Model Coefficcients')
        plt.show()
    print ("\n", "----------------LASSO REGRESSION------------------------")

    lasso = Lasso(alpha=.01, normalize=True)
    first_model_fit(lasso, test_df, predictors, target)
    coef1 = pd.Series(lasso.coef_, predictors).sort_values()
    if plotting:
        coef1.plot(kind='barh', title='Model Coefficcients')
        plt.show()
    print("\n","------------------RANDOM FOREST----------------------")
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1)
    first_model_fit(rf, test_df, predictors, target)
    rf.fit(test_df[predictors], test_df[target])
    rf.feature_importances_
    sorted_features = pd.Series(rf.feature_importances_, predictors).sort_values(ascending=False)
    if plotting:
        print("\n----------------FEATURES SORTED BY THEIR SCORE:--------------------")
        sorted_features[::-1].plot(kind='barh')

    return sorted_features[:num_feature].index.tolist()  # return top 10 features


df = df.join(pd.get_dummies(df.city))
np.sum(df.isnull(), axis=0)
ignore_feature_list = ['city', 'ndvi_ne', 'week_start_date', 'total_cases']
predictors = [feature for feature in df.columns.tolist() if feature not in ignore_feature_list]
target = 'total_cases'
df_mean = df.fillna(df.mean())
print(first_model(df_mean, predictors, target, plotting=True))

test = pd.read_csv('./dengue_features_test.csv')
test = test.join(pd.get_dummies(test.city))
print(test)


def make_prediction(algorithm, train_df, test_df, predictors, target):
    print(predictors, "\n......\n", target)
    algorithm.fit(train_df[predictors], train_df[target])
    predictions = algorithm.predict(test_df[predictors])
    result_df = test_df[['city', 'year', 'weekofyear']].copy()
    result_df['total_cases'] = predictions
    result_df.total_cases = result_df.total_cases.round()
    result_df.total_cases = result_df.total_cases.astype(int)

    return result_df


rf = RandomForestRegressor(n_estimators=300, n_jobs=-1)
result = make_prediction(rf, df, test, predictors, target)
