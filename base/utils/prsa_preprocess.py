import pandas as pd
from scipy.interpolate import lagrange
import numpy as np
import joblib


class PRSA_Preprocess(object):

    def __init__(self):
        super(PRSA_Preprocess, self).__init__()

        self.raw_df = pd.read_csv(r'..\data\raw data\PRSA_Data_Aotizhongxin_20130301-20170228.csv')
        IndexDf = pd.DataFrame()
        TempDf = pd.DataFrame()
        # IndexDf["Unnamed: 0"] = self.raw_df["Unnamed: 0"]
        # IndexDf["index"] = self.raw_df["Unnamed: 0"]
        # IndexDf["No"] = self.raw_df["Unnamed: 0"]
        TempDf = self.raw_df.iloc[:, 5:]
        # self.raw_df.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'], axis=1)
        self.raw_df = pd.concat([IndexDf,TempDf])
        # self.raw_df["Unnamed: 0"].astype("int")
        # self.raw_df["index"].astype("int")
        # self.raw_df["No"].astype("int")
        self.raw_df1 = pd.read_csv(r'..\data\pre data\PRSA_Data_Aotizhongxin_20140101-20151231.csv')
        # self.df = self.split_df()

        self.df = self.data_washing()

    # 原数据共有 35604 行数据，每隔一小时抽样生成的。本实验仅实验两年（2015, 2016）共 17544 条数据作为训练集。2016 年是闰年，会多出 24 条数据。
    def split_df(self, ):
        year_tuple = tuple(self.raw_df.groupby(['year']))

        df_2015 = year_tuple[2][1]
        df_2016 = year_tuple[3][1]

        return pd.concat([df_2015, df_2016]).reset_index()

    # 不做异常值检测处理，直接使用拉格朗日插值法填补缺失值，再向上填补缺失值
    def data_washing(self, ):
        continuous_item_list = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

        df = self.lagrange_interpolate(self.raw_df, continuous_item_list, k=8)
        df = df.fillna(method='ffill')

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

                df.iloc[n][col] = lagrange(y.index, list(y))(n)

        return df


class PRSA_Embedding(object):

    def __init__(self):
        super(PRSA_Embedding, self).__init__()

        self.df = pd.read_csv(r'..\data\raw data\PRSA_Data_Aotizhongxin_20130301-20170228.csv')
        self.df = self.df[['month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                           'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']]

        wd_dict = {'E': 0, 'ENE': 1, 'ESE': 2, 'N': 3, 'NE': 4, 'NNE': 5, 'NNW': 6, 'NW': 7, 'S': 8, 'SE': 9,
                   'SSE': 10, 'SSW': 11, 'SW': 12, 'W': 13, 'WNW': 14, 'WSW': 15}
        self.df['wd'] = self.df['wd'].map(wd_dict)

        data_array = self.df.values
        joblib.dump(data_array, '../data/data array/prsa1.array')


if __name__ == '__main__':

    prsa_preprocesser = PRSA_Preprocess()
    prsa_preprocesser.df.to_csv(r'..\data\raw data\PRSA_Data_Aotizhongxin_20130301-20170228.csv')

    prsa_embedder = PRSA_Embedding()

    data_array = joblib.load('../data/data array/prsa1.array')
