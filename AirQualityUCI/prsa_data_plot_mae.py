import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 羽毛球小论文多模型在 PRSA 数据集上的对比实验

'''
input_length = 32, input_length = 64, input_length = 96, input_length = 128

out_lengthpredict_length = 1
'''
lstm_array1 = np.array([
0.0849450,
0.0856720,
                                     0.103912,
                                     0.107601,
                                     0.108980,
                                     0.122666,
                                     0.123873,
                                     0.115239,
])
blstm_array1= np.array([
0.086045,
0.086569,
    0.088005,
    0.103857,
    0.115263,
    0.116934,
    0.124099,
    0.126013,
])
blstm_lstm_array1  = np.array([
0.093536,
0.103202,
     0.117820,
     0.117778,
     0.116157,
     0.115293,
     0.1136460,
     0.112401,
])

att_blstm_array1 = np.array([
0.091183,
0.090379,

                               0.0894470,
                               0.0924518,
                               0.0937280,
                               0.1009206,
                               0.1025116,
                               0.1065290,
    ])
bla_array1= np.array([
0.0929508,
0.0903536,
    0.089740,
    0.089768,
    0.094972,
    0.100185,
    0.096725,
    0.104395,
])
modle_list1 = [lstm_array1, blstm_array1, blstm_lstm_array1, att_blstm_array1, bla_array1, ]
# modle_list = [lstm_array, blstm_array, blstm_lstm_array]
marker_list = ['+', 'x', 'o', '*', '^']
plt.figure()
x_scale = np.arange(lstm_array1.shape[0])
handles_list = list()



for maker,model in zip(marker_list,modle_list1):
    ax = plt.plot(x_scale, model, marker=maker, linestyle='--')
    handles_list.append(ax)

plt.legend(labels=['LSTM', 'BLSTM', 'BLSTM-L', 'AttBLSTM', 'BLA'])
# plt.legend(labels=['LSTM', 'BLSTM', 'BLSTM-L'])
plt.xticks([0, 1, 2, 3,4,5,6,7], labels=["16","32","48", '64', "80", '96', '112',  '128'])
plt.xlim([2, 7])

plt.xlabel('input sequence length')
plt.ylabel('test mae')
plt.grid(axis='y', linestyle='--')

plt.title('多模型在AirQuality数据集上的测试结果 (MAE)')

plt.show()
plt.savefig('badminton model AirQuality dataset test mae1.svg', format='svg')































































































