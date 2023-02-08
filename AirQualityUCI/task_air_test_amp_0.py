import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from models import LSTM, AttBLSTM, BLSTM_L, BLSTM, BLA,LinearTransformer
from utils import Air_Dataset

'''
Typical Mixed Precision Training
'''


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train(model: nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader,
          learning_rate: float,
          epochs: int,
          input_len: int = 64,
          step_lr: bool = True,
          lr_change_step: int = 10,
          gamma: float = 0.99,
          device: str = "cuda"):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_change_step, gamma=gamma)
    loss_caculate = nn.MSELoss().to(device)
    loss_ca1 = nn.L1Loss().to(device)
    scaler = amp.GradScaler(enabled=True)

    train_loss_list = list()
    test_loss_list = list()
    test_loss = torch.Tensor([float('inf')])

    weight_file_path = './data_weight'+"_"+model.name()+"/"

    if not os.path.exists(weight_file_path):
        os.mkdir(weight_file_path)
    break_flag = 0
    with tqdm(total=epochs, desc='Model Training') as pbar:
        for epoch in range(1, epochs + 1):

            train_total_loss = 0.
            test_total_loss = 0.
            train_total_loss1 = 0.
            test_total_loss1 = 0.
            '''
            Train model
            '''
            model.train()
            loss = None

            for i in train_loader:
                sequence = i[0].to(device)
                target = i[1].squeeze(1).to(device)

                optimizer.zero_grad()

                with amp.autocast(enabled=True):
                    ouputs = model(sequence)
                    loss = loss_caculate(ouputs, target[:, -1, :])
                    loss1 = loss_ca1(ouputs, target[:, -1, :])
                scaler.scale(loss).backward()
                clip_gradient(optimizer, 10e5)
                scaler.step(optimizer)
                scaler.update()

                train_total_loss += loss.item()
                train_total_loss1 += loss1.item()

            if step_lr:
                scheduler.step()

            '''
            Evaluate model
            '''
            model.eval()
            loss = None

            with torch.no_grad():
                for i in test_loader:
                    sequence = i[0].to(device)
                    target = i[1].squeeze(1).to(device)

                    ouputs = model(sequence)
                    loss = loss_caculate(ouputs, target[:, -1, :])
                    # loss = loss_caculate(ouputs, target)
                    loss1 = loss_ca1(ouputs, target[:, -1, :])
                    test_total_loss += loss.item()
                    test_total_loss1 += loss1.item()
                if test_total_loss <= test_loss:
                    for name in os.listdir(weight_file_path):
                        file = weight_file_path + '/' + name
                        # print(int(name.split('_')[-1].split('.')[0]))
                        try:
                            if int(name.split('_')[-1]) == input_len:
                                os.remove(file)
                        except:
                            pass
                    torch.save(model.state_dict(),
                               weight_file_path + '/' + '{}_{}_{}_{}_{}.pt'.format(model.name(),input_len, test_total_loss,
                                                                                   test_total_loss1, epoch))
                    test_loss = test_total_loss
                    final_name = weight_file_path + '/' + '{}_{}_{}_{}_{}.pt'.format(model.name(),input_len,
                                                                                     test_total_loss, test_total_loss1
                                                                                     , epoch)
                    break_flag = 0
                else:
                    break_flag += 1
            # train_loss_list.append(train_total_loss)
            # joblib.dump(train_loss_list, './logs/{}_train_loss_test_amp_0.list'.format(model.name()))
            # test_loss_list.append(test_total_loss)
            # joblib.dump(test_loss_list, './logs/{}_test_loss_test_amp_0.list'.format(model.name()))

            tqdm.write(
                'Epoch: {:5} | Train Loss: {:8}| Train MAE: {:8} | Test Loss: {:8}| Test MAE: {:8}  | LR: {:8}'.format(
                    epoch, train_total_loss, train_total_loss1,
                    test_total_loss, test_total_loss1,
                    scheduler.get_last_lr()[0]))

            pbar.update(1)
            if epoch >= 350 and break_flag >= 50:
                break
    return final_name


def xtest(model: nn.Module,
         test_dataset: Air_Dataset,
         device: str = "cuda"):
    model.to(device)
    model.eval()
    loss_caculate = nn.MSELoss().to(device)
    loss_ca1 = nn.L1Loss().to(device)
    test_total_loss = 0.
    test_total_loss1 = 0.
    weight_file_path = './alldata_weight_BLA/'
    predict_array_list = list()
    target_array_list = list()

    for i in test_dataset:
        input_tensor = i[0].unsqueeze(0).to(device)
        target_tensor = i[1]

        with torch.no_grad():
            output_tensor = model(input_tensor).squeeze(0).cpu()

            loss = loss_caculate(output_tensor, target_tensor[-1])
            test_total_loss += loss.item()

            loss1 = loss_ca1(output_tensor, target_tensor[-1])
            test_total_loss1 += loss1.item()

            predict_array_list.append(output_tensor.numpy())
            target_array_list.append(target_tensor[-1].numpy())

    input_array = test_dataset[0][0].cpu().numpy()
    predict_array = np.array(predict_array_list)
    target_array = np.array(target_array_list)

    return input_array.T, predict_array.T, target_array.T, test_total_loss, test_total_loss1, len(test_dataset)


if __name__ == '__main__':
    file_path = './data/AirQualityUCI/data_array/air.array'

    batch_size = 1024
    do_train = True
    learning_rate = 0.0001
    epochs = 4000

    do_test = False
    input_lens = [32, 48, 64, 80, 96, 112]
    # 定义模型时需要确定是单一变量预测还是多变量预测
    # model = BLA(input_size=12, output_size=12)
    # model = BLSTM(i_size=12, o_size=12)

    # model = BLSTM_L(input_size=12, output_size=12)
    # model = AttBLSTM(input_size=12, output_size=12)
    model = LinearTransformer(input_size=12, output_size=12)
    for input_len in input_lens:
        print(f"输入长度为{input_len}")
        train_dataset = Air_Dataset(input_len=input_len, train=True, file_path=file_path, transformer=True)
        test_dataset = Air_Dataset(input_len=input_len, train=False, file_path=file_path, transformer=True)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

        start = time.time()
        final_name = "./data_weight/LSTM_142_0.06178278289735317_0.207856185734272_16.pt"
        if do_train:
            final_name = train(model=model, train_loader=train_loader, test_loader=test_loader, input_len=input_len,
                               learning_rate=learning_rate, epochs=epochs)

        end = time.time()

        # print(end - start)

        if do_test:
            # model = BLSTM(input_size=12, output_size=12)
            model.load_state_dict(torch.load(final_name))
            i, p, t, loss_MSE, loss_MAE, length = xtest(model=model, test_dataset=test_dataset)

            # for plot_feature_idx in range(0, 15):
            #     plt.figure(plot_feature_idx)
            #     plt.plot(range(0, i.shape[1]), i[plot_feature_idx], )
            #     plt.plot(range(i.shape[1], i.shape[1] + p.shape[1]), p[plot_feature_idx], )
            #     plt.plot(range(i.shape[1], i.shape[1] + p.shape[1]), t[plot_feature_idx], )

            # plt.show()

            print('MSE: {}'.format(loss_MSE))
            print('MAE: {}'.format(loss_MAE))
