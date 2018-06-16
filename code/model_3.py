import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from code import model_2_preprocess

sj_train, iq_train, sj_label, iq_label = model_2_preprocess.preprocess('../data/dengue_features_train.csv',
                                                                       '../data/dengue_labels_train.csv')

sj_x_train, sj_x_test, sj_y_train, sj_y_test = train_test_split(sj_train, sj_label['total_cases'], test_size=0.33,
                                                                random_state=0, shuffle=False)

iq_x_train, iq_x_test, iq_y_train, iq_y_test = train_test_split(iq_train, iq_label['total_cases'], test_size=0.33,
                                                                random_state=0, shuffle=False)

sj_model = RandomForestRegressor(n_estimators=840, max_depth=65, max_features=10, min_samples_split=70,
                                 min_samples_leaf=3, criterion='mae', warm_start=True)
sj_model.fit(sj_x_train, sj_y_train)
sj_predict_value = sj_model.predict(sj_x_test)


iq_model = RandomForestRegressor(n_estimators=751, max_features='auto', max_depth=76, min_samples_leaf=45,
                                 criterion='mae', min_weight_fraction_leaf=0.1, warm_start=True, min_samples_split=4)
iq_model.fit(iq_x_train, iq_y_train)
iq_predict_value = iq_model.predict(iq_x_test)

print("SJ" + str(mean_absolute_error(sj_y_test, sj_predict_value)))
print("IQ " + str(mean_absolute_error(iq_y_test, iq_predict_value)))

sj_test, iq_test, sj_test_label, iq_test_label = model_2_preprocess.preprocess('../data/dengue_features_test.csv')

sj_predictions = sj_model.predict(sj_test).astype(int)
iq_predictions = iq_model.predict(iq_test).astype(int)

submission = pd.read_csv("../data/submission_format.csv")
submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("../result/submission_3.csv")

