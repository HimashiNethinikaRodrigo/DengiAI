import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE, mutual_info_regression, mutual_info_classif, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.testing import set_random_state

feature = pd.read_csv('../data/dengue_features_train.csv')
label = pd.read_csv('../data/dengue_labels_train.csv')
df = pd.merge(feature, label, how='outer', on=label.columns.tolist()[:-1])

np.sum(df.isnull(), axis=0)
df = df.dropna(axis=0, thresh=20)
np.sum(df.isnull(), axis=0)
df.city.value_counts()
df.fillna(method='ffill', inplace=True)
df = df.fillna(df.mean())

df_sj = df[df.city == 'sj'].copy()
df_iq = df[df.city == 'iq'].copy()
df = df.drop(["city", "year", "weekofyear", "week_start_date"], axis=1)
x_set = df.drop(["total_cases"], axis=1)
y_set = df["total_cases"]
print(x_set.columns.tolist())

# # Feature selection from Logistic Regression RFE
# model = LogisticRegression()
# rfe = RFE(model, 5)
# fit = rfe.fit(x_set, y_set)
# print(fit)
# print("Num Features: ", fit.n_features_)
# print("Selected Features: ", fit.support_)
# print("Feature Ranking: ", fit.ranking_)
#
# # Importance of features
# model = ExtraTreesClassifier()
# model.fit(x_set, y_set)
# print(model.feature_importances_)

# mutual_info_classif
# mutual = mutual_info_classif(x_set, y_set, n_neighbors=2, random_state=5)
# print(mutual)

# L1 based feature selection
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_set, y_set)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(x_set)
# print(x_set.shape)
# print(X_new.shape)

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', RandomForestClassifier())
])
print(clf.fit(x_set, y_set))