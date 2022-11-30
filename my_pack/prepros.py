from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd


def data_prepro_StdScaler(train, data):
    list_num = []
    list_cat = []
    for i in data.columns.tolist():
        if 'float64' in str(data[i].dtype) or 'int64' in str(data[i].dtype):
            list_num.append(i)
        else:
            list_cat.append(i)

    scaler = StandardScaler()
    scaler.fit(train[list_num])
    data_scaled = scaler.transform(data[list_num])
    data_scaled_df = pd.DataFrame(data_scaled)
    data_scaled_df.columns = list_num

    ohe = OneHotEncoder()
    data_ohe = ohe.fit_transform(data[list_cat])
    data_ohe_df = pd.DataFrame.sparse.from_spmatrix(data_ohe)

    data_prep_df = pd.concat([data_ohe_df, data_scaled_df], axis=1)

    return data_prep_df

def data_prepro_MinMaxScaler(train, data):
    list_num = []
    list_cat = []
    for i in data.columns.tolist():
        if 'float64' in str(data[i].dtype) or 'int64' in str(data[i].dtype):
            list_num.append(i)
        else:
            list_cat.append(i)

    scaler = MinMaxScaler()
    scaler.fit(train[list_num])
    data_scaled = scaler.transform(data[list_num])
    data_scaled_df = pd.DataFrame(data_scaled)
    data_scaled_df.columns = list_num

    ohe = OneHotEncoder()
    data_ohe = ohe.fit_transform(data[list_cat])
    data_ohe_df = pd.DataFrame.sparse.from_spmatrix(data_ohe)

    data_prep_df = pd.concat([data_ohe_df, data_scaled_df], axis=1)

    return data_prep_df

def Hello():
    print('Yeah!!!')
