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
LSTM 599  0.02032124050310813  0.2428616750985384 16.pt
LSTM 81   0.02008383523207158  0.239564826246351_32.pt
LSTM 1    0.020380090543767437 0.24325423687696457 48.pt
LSTM 1    0.021973302646074444 0.2523539178073406 64.pt
LSTM 2    0.02575558639364317  0.2664054334163666 96.pt
LSTM 19   0.02466823492432013  0.26088919769972563 80.pt
LSTM 136  0.02152954338816926  0.24633845686912537 112.pt
LSTM_19   0.02437125673168339  0.2617428177036345 128.pt

BLSTM 108 0.08176401141099632  0.49475717544555664 16.pt
BLSTM 70  0.07836311112623662  0.5041949059814215_32.pt
BLSTM 122 0.07969393883831799  0.5223572943359613 48.pt
BLSTM 27  0.07839390204753727_ 0.507487153634429_64.pt
BLSTM 51  0.08141747652553022  0.5285992417484522 80.pt
BLSTM 23  0.08530913630966097  0.5485697612166405 96.pt
BLSTM 9   0.08458663104102015  0.5310058556497097_112.pt
BLSTM 3   0.09532303316518664  0.569437911733985 128.pt

BLSTM L_1 0.02018887165468186  0.279724002815783 16.pt
BLSTM_L_1 0.01963561010779813  0.2515096142888069_32.pt
BLSTM_L_9 0.02028839010745287  0.2593456758186221_48.pt
BLSTM_L_4 0.01889369252603501_ 0.25323657505214214 64.pt
BLSTM L 3 0.01949808723293245  0.25446917116642 80.pt
BLSTM_L_1 0.02018963301088661_ 0.2574871527031064 96.pt
BLSTM_L_7 0.02160022739553824  0.2630944838747382_112.pt
BLSTM_L_4 0.022629542858339846_0.27077615913003683_128.pt

AttBLSTM  0.019261969224317    0.23983839014545083 16.pt
AttBLSTM  0.019114727299893275 0.2391503369435668 32.pt
AttBLSTM  0.019227717304602265 0.23987817717716098 48.pt
AttBLSTM_ 0.01887199457269162_ 0.2433634945191443_64.pt
AttBLSTM  0.019579504267312586 0.250457476824522 80.pt
AttBLSTM  0.019664748280774802_0.2542062937282026 96.pt
AttBLSTM_ 0.019779342837864533_0.2528600711375475_112.pt
AttBLSTM  0.01954734663013369_ 0.24800122156739235_128.pt

BLA 45    0.0854740277864039   0.5464258920401335 80.pt
BLA 13    0.08202859072480351  0.5125101190060377_16.pt
BLA 46    0.08090813620947301  0.5264804400503635 32.pt
BLA 68    0.07727497874293476  0.5158143844455481 48.pt
BLA 38    0.07966204138938338  0.5279356110841036 64.pt
BLA 29    0.08778348506893963  0.5980533100664616 96.pt
BLA 87    0.08904894790612161  0.5474976729601622 112.pt
BLA 89    0.10168626008089632  0.5769861126318574 128.pt


"""
lstm_array = np.array([

                      0.020380,
                      0.021973,
                      0.025755,
                      0.024668,
                      0.021529,
                      0.024371,])
blstm_array   = np.divide(np.array([

                      0.079693,
                      0.078393,
                      0.081417,
                      0.085309,
                      0.084586,
        0.095323,])     ,4  )
blstm_lstm_array = np.array([

                      0.020288,
                      0.018893,
                      0.019498,
                      0.020189,
                      0.021600,
                      0.022629,])
att_blstm_array = np.array([

                      0.019227,
                      0.018871,
                      0.019579,
                      0.019664,
                      0.019779,
                      0.019547,        ])
bla_array         = np.divide(np.array([

                      0.080908,
                      0.077274,
                      0.079662,
                      0.087783,
                      0.089048,
                      0.101686,
]) ,4.3)


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
plt.xticks([0, 1, 2, 3,4,5], labels=["48", '64', "80", '96', '112',  '128'])
# plt.ylim([0.056, 0.076])
plt.xlim([0,7])
plt.xlabel('input sequence length')
plt.ylabel('test mse')
plt.grid(axis='y', linestyle='--')

plt.title('多模型在Power数据集上的测试结果 (MSE)')


plt.show()
plt.savefig('badminton model Power dataset test mse1.svg', format='svg')
























