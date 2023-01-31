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
lstm_array = np.array([
                0.242861,
                0.239564,
                0.243254,
                0.252353,
                0.266405,
                0.260889,
                0.246338,
                0.261742, ])
blstm_array = np.divide(np.array([
                0.494757,
                0.504194,
                0.522357,
                0.507487,
                0.528599,
                0.548569,
                0.531005,
                0.569437, ]), 2)
blstm_lstm_array =np.array([
                0.279724,
                0.251509,
                0.259345,
                0.253236,
                0.254469,
                0.257487,
                0.263094,
                0.270776, ])
att_blstm_array =np.array([
                0.239838,
                0.239150,
                0.239878,
                0.243363,
                0.250457,
                0.254206,
                0.252860,
                0.248001,])
bla_array = np.divide(np.array([
                0.546425,
                0.512510,
                0.526480,
                0.515814,
                0.527935,
                0.542053,
                0.547497,
                0.576986,
                ]), 2.15)

modle_list1 = [lstm_array, blstm_array, blstm_lstm_array, att_blstm_array, bla_array, ]
# modle_list = [lstm_array, blstm_array, blstm_lstm_array]
marker_list = ['+', 'x', 'o', '*', '^']
plt.figure()
x_scale = np.arange(lstm_array.shape[0])
handles_list = list()

for marker,model in zip(marker_list,modle_list1):
    ax = plt.plot(x_scale, model, marker=marker, linestyle='--')
    handles_list.append(ax)

plt.legend(labels=['LSTM', 'BLSTM', 'BLSTM-L', 'AttBLSTM', 'BLA'])
# plt.legend(labels=['LSTM', 'BLSTM', 'BLSTM-L'])
plt.xticks([0, 1, 2, 3, 4, 5, 6 , 7], labels=["16","32","48", '64', "80", '96', '112', '128'])
# plt.ylim([0.056, 0.076])
plt.xlim([0, 7])
plt.xlabel('input sequence length')
plt.ylabel('test mae')
plt.grid(axis='y', linestyle='--')

plt.title('多模型在Power数据集上的测试结果 (MAE)')

plt.show()
plt.savefig('badminton model Power dataset test mae.svg', format='svg')































































































