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
"""
LSTM   0.01601366326212883  0.08494508266448975_16.pt
LSTM 7 0.016000233590602875 0.08567200601100922 32.pt
LSTM 6 0.022865867242217064 0.10391249507665634 48.pt
LSTM 9 0.024201324209570885 0.1076018214225769_64.pt
LSTM 9 0.025173833593726158 0.1089807003736496 80.pt
LSTM 4 0.031108831986784935 0.12266622483730316 96.pt
LSTM 5 0.031974658370018005 0.12387307733297348 112.pt
LSTM 8 0.027501625940203667 0.1152399331331253 128.pt

BLSTM   0.016418669372797012  0.0860452875494957 16.pt
BLSTM 1 0.016145668923854828  0.08656978607177734 32.pt
BLSTM 4 0.01718805357813835   0.08800505846738815 48.pt
BLSTM 3 0.02273450419306755   0.10385745763778687_64pt
BLSTM 2 0.027125608175992966  0.11526308953762054 80.pt
BLSTM 2 0.027694862335920334  0.11693474650382996 96.pt
BLSTM_1 0.030278179794549942  0.12409916520118713_112.pt
BLSTM 2 0.031183145940303802  0.12601327896118164 128.

BLSTM L 68  0.019313601776957512 0.09353630989789963 16.pt
BLSTM L 5   0.023132771253585815 0.10320232808589935 32.pt
BLSTM L 4   0.028187252581119537 0.11782067269086838 48.pt
BLSTM L 299 0.028215065598487854 0.1177787184715271 64.pt
BLSTM L 353 0.027561485767364502 0.11615707725286484 80.pt 
BLSTM L 249 0.02724510431289673  0.11529331654310226 96.pt
BLST 1 444  0.026564210653305054 0.1136460155248642 112pt
BLSTM L 310 0.025598032400012016 0.11240129172801971 128.pt

AttBLSTM 581 0.01827574335038662  0.09118352830410004 16.pt
AttBLSTM 165 0.018138689920306206 0.09037991613149643 32.pt
AttBLSTM 8   0.018106698989868164 0.089447021484375 48.pt
AttBLSTM 10  0.019176602363586426 0.09245189279317856 64.pt
AttBLSTM 10  0.01943674311041832  0.09372800588607788 80.pt
AttBLSTM 5   0.022586658596992493 0.10092063993215561_96.pt
AttBLSTM 150 0.023407503962516785 0.10251166671514511_112.pt
AttBLSTM 3   0.024639016017317772 0.10652907937765121_128.pt

BLA 296 0.019344549626111984 0.09295082092285156_16.pt
BLA 138 0.01799974963068962  0.09035369008779526_32.pt
BLA 22_ 0.018209774047136307 0.08974029868841171_48.pt
BLA 8   0.0181921124458313   0.08976814150810242_64.pt
BLA 5_  0.019961722195148468 0.09497283399105072 80.pt
BLA 3   0.021838422864675522 0.10018544644117355 112.pt
BLA 3   0.020990677177906036 0.09672503173351288 96.pt
BLA 2   0.023515760898590088 0.1043958067893982 128.pt




"""
lstm_array = np.array([
                              0.016013,
                              0.016000,
                              0.022865,
                              0.024201,
                              0.025173,
                              0.031108,
                              0.031974,
                              0.027501,
])

blstm_array   = np.array([
    0.016418,
    0.016145,
    0.0171880,
    0.0227345,
    0.0271256,
    0.0276948,
    0.0302781,
    0.03118314
])
blstm_lstm_array = np.array([
    0.019313,
    0.023132,
    0.0281872,
    0.0282150,
    0.0275614,
    0.0272451,
    0.0265642,
    0.0255980,
])
att_blstm_array = np.array([
                              0.018275,
                              0.018138,
                              0.018106,
                              0.019176,
                              0.019436,
                              0.022586,
                              0.023407,
                              0.024639,
    ])



bla_array         = np.array([
    0.019344,
    0.017999,
    0.0182097,
    0.0181921,
    0.0199617,
    0.0218384,
    0.0209906,
    0.0235157,
])


# transformer_array = np.array([0.060054, 0.059045, 0.058098, 0.057998])

modle_list = [lstm_array, blstm_array, blstm_lstm_array, att_blstm_array, bla_array, ]
# modle_list = [lstm_array, blstm_array, blstm_lstm_array]
marker_list = ['+', 'x', 'o', '*', '^']
plt.figure()
x_scale = np.arange(lstm_array.shape[0])
handles_list = list()

for maker,model in zip(marker_list,modle_list):
    ax = plt.plot(x_scale, model, marker=maker, linestyle='--')
    handles_list.append(ax)

plt.legend(labels=['LSTM', 'BLSTM', 'BLSTM-L', 'AttBLSTM', 'BLA'])
# plt.legend(labels=['LSTM', 'BLSTM', 'BLSTM-L'])
plt.xticks([0, 1, 2, 3,4,5,6,7], labels=["16","32","48", '64', "80", '96', '112',  '128'])
plt.xlim([0, 7])
plt.xlabel('input sequence length')
plt.ylabel('test mse')
plt.grid(axis='y', linestyle='--')

plt.title('多模型在AirQuality数据集上的测试结果 (MSE)')


plt.show()
plt.savefig('badminton model AirQuality dataset test mse1.svg', format='svg')
























