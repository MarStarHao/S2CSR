
from S2CSD.adapers import S2CSD_Adaper
from S2CSD.default_params import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from TSpy.view import plot_mts
import matplotlib.pyplot as plt
from TSpy.utils import *
import os
import numpy as np
import torch
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from S2CSD.adapers import S2CSD_Adaper
from S2CSD.clustering import DPGMM
#from S2CSD.params import *
from S2CSD.s2csd import S2CSD
"""
这个部分是论文 图1  Time series segmentation for state inference.
"""


win_size = 500 # 论文中的窗口大小 参数w  从demo的图中显示 win_size对状态的分割具有明显的影响，越大分割序列越粗，越小分割的序列越细。这种设定还是要根据数据本身的特点设定
step = 50 # 论文中的步长 s  从demo的运行时长 可以看出，该参数和算法的运行时间相关

# Load data
data = pd.read_csv('data/MoCap/4d/amc_86_07.4d', sep=' ').to_numpy() # 转成一个numpy的数据
#data = pd.read_csv('data/UCR-SEG/Cane_100_2345.txt').to_numpy() # 转成一个numpy的数据
# 通过去除平均值和缩放到单位方差来标准化特征
data = StandardScaler().fit_transform(data)
#print(data.shape)
# model 参数
params_LSE['in_channels'] = data.shape[1] # 通道数量 4
params_LSE['out_channels'] = 4
params_LSE['nb_steps'] = 30 #
params_LSE['win_size'] = win_size
params_LSE['win_type'] = 'hanning' # {rect, hanning} 窗口类型
# DPGMM 是在嵌入空间中的聚类模型
# t2s的对象
s2c = S2CSD(win_size, step, S2CSD_Adaper(params_LSE), DPGMM(None))
s2c.fit(data, win_size, step) #模型训练
#
plt.style.use('classic')
plot_mts(data,s2c.state_seq) # 输出了状态标签
plt.show()
plt.savefig("./Image/amc_86_07.4d.pdf")
#plt.savefig()
#plt.savefig('plt.png')
#print("dada")
plt.show()


