import pandas as pd
import numpy as np
import joblib
import os


class Air_Preprocess(object):

    def __init__(self, file_path: str = '../data/AirQualityUCI/raw_data/AirQualityUCI.xlsx'):
        super(Air_Preprocess, self).__init__()

        # 若 data_array 文件夹不存在则创建
        data_array_path = '../data/AirQualityUCI/data_array/'
        if not os.path.exists(data_array_path):
            os.mkdir(data_array_path)

        self.raw_df = pd.read_excel(file_path)
        self.df = self.data_washing()
        joblib.dump(self.df, data_array_path + 'air.array')

    def data_washing(self, ):
        self.raw_df.replace(-200, np.nan, inplace=True)  # 将数据集中的无效值（-200）替换为空值 nan
        self.raw_df = self.raw_df.sort_values(by=['Date', 'Time'], ascending=True)  # 将数据根据 Data 和 Time 升序排列

        item_list = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
                     'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
        self.raw_df = self.raw_df[item_list]  # 保留 item_list 中的特征
        """此时的统计信息
        Int64Index: 9357 entries, 0 to 9356
        Data columns (total 13 columns):
         #   Column         Non-Null Count  Dtype  
        ---  ------         --------------  -----
         0   CO(GT)         7674 non-null   float64
         1   PT08.S1(CO)    8991 non-null   float64
         2   NMHC(GT)       914 non-null    float64
         3   C6H6(GT)       8991 non-null   float64
         4   PT08.S2(NMHC)  8991 non-null   float64
         5   NOx(GT)        7718 non-null   float64
         6   PT08.S3(NOx)   8991 non-null   float64
         7   NO2(GT)        7715 non-null   float64
         8   PT08.S4(NO2)   8991 non-null   float64
         9   PT08.S5(O3)    8991 non-null   float64
         10  T              8991 non-null   float64
         11  RH             8991 non-null   float64
         12  AH             8991 non-null   float64
        """
        self.raw_df.drop(['NMHC(GT)'], axis=1, inplace=True)  # 删掉只有914个数据的 'NMHC(GT)' 列
        self.raw_df = self.raw_df.fillna(method='ffill')  # 向上填补缺失值
        df = self.raw_df.values

        return df


if __name__ == '__main__':
    air_preprocesser = Air_Preprocess()
    file_path = '../data/AirQualityUCI/data_array/air.array'
    air_array1 = joblib.load(file_path)
    print(len(air_array1))
