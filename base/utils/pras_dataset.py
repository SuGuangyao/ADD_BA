from torch.utils.data import Dataset
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch


class PRAS_Dataset(Dataset):

    def __init__(self,
                 input_len: int = 64,
                 output_len: int = 1,
                 train: bool = True,
                 transformer: bool = False,
                 train_test_split_ratio: float = 0.8,
                 file_path: str = '../data/data array/prsa.array'):
        super(PRAS_Dataset, self).__init__()

        self.prsa_array = joblib.load(file_path)

        self.train_array = self.prsa_array[:int(self.prsa_array.shape[0] * train_test_split_ratio)]

        self.test_array = self.prsa_array[int(self.prsa_array.shape[0] * train_test_split_ratio):]

        # 注意训练集和测试集要分开做归一化，否则会造成数据泄露
        self.train_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.test_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.tranformed_train_array = self.get_transform(self.train_scaler, self.train_array)
        self.tranformed_test_array = self.get_transform(self.test_scaler, self.test_array)

        self.sequence_list = list()
        self.target_list = list()
        
        if not transformer:
            if train:
                for i in range(self.tranformed_train_array.shape[0] - input_len - output_len + 1):
                    self.sequence_list.append(self.tranformed_train_array[i: i+input_len])
                    self.target_list.append(self.tranformed_train_array[i+input_len: i+input_len+output_len])
            else:
                for i in range(self.tranformed_test_array.shape[0] - input_len - output_len + 1):
                    self.sequence_list.append(self.tranformed_test_array[i: i+input_len])
                    self.target_list.append(self.tranformed_test_array[i + input_len: i + input_len + output_len])
        else:
            if train:
                for i in range(self.tranformed_train_array.shape[0] - input_len - output_len + 1):
                    self.sequence_list.append(self.tranformed_train_array[i: i+input_len])
                    self.target_list.append(self.tranformed_train_array[i+output_len: i+input_len+output_len])
            else:
                for i in range(self.tranformed_test_array.shape[0] - input_len - output_len + 1):
                    self.sequence_list.append(self.tranformed_test_array[i: i+input_len])
                    self.target_list.append(self.tranformed_test_array[i + output_len: i + input_len + output_len])


    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequence_list[idx], dtype=torch.float32)
        target = torch.tensor(self.target_list[idx], dtype=torch.float32)

        return sequence, target

    def __len__(self):
        return len(self.sequence_list)

    def get_transform(self, scaler: MinMaxScaler, array: np.array):
        return scaler.fit_transform(array)

    def get_inverse_transfoerm(self, scaler: MinMaxScaler, array: np.array):
        return scaler.inverse_transform(array)


if __name__ == '__main__':
    train_dataset = PRAS_Dataset(train=True)
    test_dataset = PRAS_Dataset(train=False)
