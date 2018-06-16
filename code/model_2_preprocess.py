import pandas as pd


def preprocess(data_file, labels_file=None):
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c', 'precipitation_amt_mm', 'week_start_date']
    df = pd.read_csv(data_file, index_col=[0, 1, 2])

    df['station_avg_temp_c_mv_avg'] = df['station_avg_temp_c'].rolling(window=5).mean()
    df['precipitation_amt_mm_mv_avg'] = df['precipitation_amt_mm'].rolling(window=5).mean()
    features.append('station_avg_temp_c_mv_avg')
    features.append('precipitation_amt_mm_mv_avg')

    df.fillna(method='ffill', inplace=True)
    df = df.fillna(df.mean())

    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    for i in range(1, 5):
        df['quarter_' + str(i)] = df['week_start_date'].apply(lambda date: 1 if (
                ((i - 1) * 3 < date.to_datetime().month) and (date.to_datetime().month <= i * 3)) else 0)
        features.append('quarter_' + str(i))

    df = df.drop(['week_start_date'], axis=1)
    features.remove('week_start_date')
    df = df[features]
    sj_label = None
    iq_label = None
    if labels_file:
        labels = pd.read_csv(labels_file, index_col=[0, 1, 2]).loc[df.index]
        sj_label = pd.DataFrame(labels.loc['sj'])
        iq_label = pd.DataFrame(labels.loc['iq'])

    sj = pd.DataFrame(df.loc['sj'])
    iq = pd.DataFrame(df.loc['iq'])

    return sj, iq, sj_label, iq_label
