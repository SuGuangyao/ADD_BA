""""
  txt process

"""
import joblib
from scipy.interpolate import lagrange
import numpy as np
import pandas as pd


class Preprocess(object):

    def __init__(self,path, filename):
        super(Preprocess, self).__init__()

        dataset = pd.read_csv(path + '/' + file_name, sep=';', header=0,
                              low_memory=False, infer_datetime_format=True, engine='c')
        dataset.replace('?', np.nan, inplace=True)
        dataset = dataset.drop(["Date","Time"], axis=1)
        values = dataset.values.astype('float32')
        # dataset.values = dataset.values.astype('float32')

        dataset['sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])
        self.raw_df = dataset

        self.df = self.data_washing()

    # 不做异常值检测处理，直接使用拉格朗日插值法填补缺失值，再向上填补缺失值
    def data_washing(self, ):
        continuous_item_list = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'sub_metering_4']
        # df = self.lagrange_interpolate(self.raw_df, continuous_item_list, k=8)
        df = self.raw_df.fillna(method='ffill')
        print(self.raw_df.shape)
        # df =self.raw_df.dropna()
        print(df.shape)
        return df

    @staticmethod
    def lagrange_interpolate(df, item_list, k=5):
        df = df.copy(deep=True)

        for col in item_list:
            col_data = df[col]
            col_nan = col_data[col_data.isnull()]
            col_nan_list = col_nan.index

            for n in col_nan_list:
                    list1 = list(range(n - k, n))

                    if list1[0] < col_data.index[0]:
                        list1 = list(range(col_data.index[0], n))

                    list2 = col_data[n:]
                    list2 = list2[list2.notnull()]
                    list2 = list(list2[0:k].index)
                    y = col_data[list1 + list2]
                    fly = np.array(list(y)).astype("float32").tolist()
                    a = lagrange(y.index, fly)(n)
                    df.iloc[n][col] = lagrange(y.index, y)(n)

        return df


if __name__ == "__main__":
    path = './raw_data'
    file_name = "household_power_consumption.txt"
    preprocess = Preprocess(path, filename=file_name)
    data_array = preprocess.df.values
    joblib.dump(data_array, './pre_data/power.array')
