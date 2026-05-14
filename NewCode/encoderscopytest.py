import numpy
import sys
import os
import losses
import networks
import numpy as np
from S2CSD.augmentations import DataTransform_T, DataTransform_F
from torch.fft import fft


sys.path.append(os.path.dirname(__file__))
import utils
import math
import torch
from losses.TFC_loss import TFCLoss
from losses.TFC_loss import TimeDomainLoss_t
from losses.TFC_loss import FrequencyDomainLoss_f
from networks.CausalCnn import CausalCNNEncoder
from networks.ConvEncoder_f import ConvBlockEncoder_f
from networks.LSTMEncoder_t import LSTMEncoder


def hanning_numpy(X):
    length = X.shape[2] # 列
    weight = (1-np.cos(2*math.pi* np.arange(length)/length))/2
    # weight = np.cos(2*math.pi*np.arange(length)/length)+0.5
    return weight*X

def hanning_tensor(X):
    length = X.size(2)
    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2
    weight = torch.tensor(weight)
    return weight.cuda()*X

class BasicEncoderClass():
    def encode(self, X):
        """
        对输入数据进行编码的基础方法，具体实现由子类完成。

        :param X: 输入数据
        """
        pass

    def save(self, X):
        """
        保存编码器相关数据的基础方法，具体实现由子类完成。

        :param X: 待保存的数据
        """
        pass

    def load(self, X):
        """
        加载编码器相关数据的基础方法，具体实现由子类完成。

        :param X: 待加载的数据
        """
        pass


class CausalConv(BasicEncoderClass):  #
    def __init__(self, win_size, batch_size, nb_steps, lr,
                 channels, depth, reduced_size, out_channels, kernel_size,
                 in_channels, cuda, gpu, M, N, win_type):
        # 时域LSTM参数
        # self.feature_dim = 256  # 特征维度
        self.hidden_size = 1024  # 隐藏层大小
        self.output_size = 512  # 输出大小
        self.input_size = 4  # 输入大小
        self.dropout_rate = 0.10  # 丢弃率
        self.num_layers = 2  # 双向 LSTM 的层数
        # 频域参数
        # self.input_channels = 4  # 单变量时间序列 (UTS) 卷积网络的输入通道数
        # self.kernel_size = 8  # 卷积核大小
        # self.stride = 1  # 卷积步长
        # self.output_channels = 4  # 卷积网络的输出通道数
        # # self.num_classes = None  # 类别数量
        # self.dropout = 0.30  # 丢弃率
        # self.batchsize = 1  #

        # 创建一个网络模型
        self.network = self.__create_network(in_channels, channels, depth,
                                             reduced_size, out_channels, kernel_size, cuda, gpu)

        self.lstmEncoder_t = networks.LSTMEncoder_t.LSTMEncoder(self.input_size, self.hidden_size,
                                                                self.num_layers, self.output_size,
                                                                self.dropout_rate)

        self.convBlockEncoder_f = networks.ConvEncoder_f.ConvBlockEncoder_f(in_channels,
                                                                            out_channels,
                                                                           kernel_size)

        self.win_type = win_type
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        # ----------------------------------------------------------
        # -----------------------------------
        self.jitter_scale_ratio = 1.1  # 抖动缩放比例
        self.jitter_ratio = 0.8  # 抖动比例
        self.max_seg = 2  # 最大分段数

        # 频域增强参数
        self.remove_frequency_ratio = 0.1  # 在频域上移除频率的比例
        self.add_frequency_ratio = 0.1  # 在频域上添加频率的比例

        # 时域LSTM参数
        # self.feature_dim = 256  # 特征维度
        # self.hidden_size = 1024  # 隐藏层大小
        # self.output_size = 512  # 输出大小
        #
        # self.dropout_rate = 0.10  # 丢弃率
        # self.num_layers = 2  # 双向 LSTM 的层数

        # 频域参数
        # ------频域编码器参数------
        self.input_channels = 4  # 单变量时间序列 (UTS) 卷积网络的输入通道数
        self.kernel_size = 8  # 卷积核大小
        self.stride = 1  # 卷积步长
        self.output_channels = 4  # 卷积网络的输出通道数
        # self.num_classes = None  # 类别数量
        self.dropout = 0.30  # 丢弃率
        self.batchsize = 1  #

        #------------------------------------------------------
        # 损失函数  这是LSE_loss.py中 实现的LSE-loss函数
        self.loss = losses.TFC_loss.TFCLoss(
            win_size, M, N, win_type
        )  # 损失函数

        self.loss_t =losses.TFC_loss.TimeDomainLoss_t(device="cuda")

        self.loss_f =losses.TFC_loss.FrequencyDomainLoss_ftest(device="cuda")


        # ----------------------------------------------------------------
        # if cuda:
        #     self.lstmEncoder_t.cuda(gpu)
        #     self.convBlockEncoder_f.cuda(gpu)
        if cuda:
            #self.lstmEncoder_t.cuda(gpu)  # 将 LSTM 编码器移到 GPU
            self.convBlockEncoder_f.cuda(gpu)  # 将频域卷积编码器移到 GPU
        #self.lstmEncoder_t.float()  # 统一为 float 类型（与输入一致）
        self.convBlockEncoder_f.float()
        # 优化器：合并主网络和辅助编码器的参数
        all_params = list(self.network.parameters()) + list(self.lstmEncoder_t.parameters()) + list(
            self.convBlockEncoder_f.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    # 创建网络模型, 创建因果卷积网络
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        # -------------------------------------------------

        # 使用因果卷积编码器创建了一个网络模型
        network = networks.CausalCnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )

        network.double()  # 将数据类型设置为双精度浮点数
        if cuda:
            network.cuda(gpu)
        return network

    # 返回一个训练network网络
    # 模型训练方法  使用给定的训练数据无监督地训练编码器。
    def fit(self, X, y=None, save_memory=False, verbose=False):
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        train = torch.from_numpy(X)  # 从numpy数组创建一个PyTorch张量
        if self.cuda:
            train = train.cuda(self.gpu)  # 使用GPU

        # -------------------------------------------------------

        # 训练数据集
        # 创建一个PyTorch数据集对象，用于加载训练数据集
        train_torch_dataset = utils.Dataset(X)  # PyTorch wrapper for a numpy dataset. 是一个numpy dataset对象
        # 用于创建数据加载器（data loader）的函数。数据加载器用于加载训练和测试数据集，并将数据划分为小批量进行处理
        # 可以方便地进行数据批量处理、乱序加载和并行读取
        #
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )  # 加载训练数据集

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            # ---------------------------------------
            for batch in train_generator:  # train_generator中的批量数据
                if self.cuda:
                    batch = batch.cuda(self.gpu)  # 使用GPU
                # --------------------------------
                # 下面部分是训练模型
                self.optimizer.zero_grad()
                # 损失函数 预测和真实标签之间的损失
                # losses.LSE_loss.LSELoss()
                loss = self.loss(batch, self.network, save_memory=save_memory)


               #------------
                batch_np = batch.cpu().numpy() if self.cuda else batch.numpy()

                X_aug_t = DataTransform_T(batch_np, self.jitter_ratio, self.jitter_scale_ratio,
                                          self.max_seg)  # 直接处理 batch（已转 tensor）
               # print("X_aug_t", batch.shape)
                z_i_t = self.lstmEncoder_t(batch.cpu().numpy())  # 原始数据嵌入（LSTM 输出）
                z_i_t_aug = self.lstmEncoder_t(X_aug_t)  # 增强数据嵌入
                z_i_t = z_i_t.unsqueeze(1).expand(z_i_t_aug.shape[0],  z_i_t_aug.shape[1], z_i_t_aug.shape[2])
                z_i_t = z_i_t.permute(1, 0, 2)
                z_i_t_aug= z_i_t_aug.permute(1, 0, 2)
                print("z_i_f_aug",z_i_t_aug.shape, z_i_t.shape)
                loss_t = self.loss_t(z_i_t, z_i_t_aug)

                # self.x_data_f = torch.fft.fft(torch.from_numpy(X)).abs()  # torch.Size([1, 4, 5673])
                #
                # self.X_aug_t = DataTransform_T(batch, self.jitter_ratio, self.jitter_scale_ratio, self.max_seg)
                #
                # self.X_data_f, self.X_aug_f = DataTransform_F(self.x_data_f, self.remove_frequency_ratio, self.add_frequency_ratio)
                # 频域增强与对比损失
                x_data_f = torch.fft.fft(batch).abs()  # 直接对 batch 做傅里叶变换
                X_data_f, X_aug_f = DataTransform_F(x_data_f, self.remove_frequency_ratio, self.add_frequency_ratio)
                z_i_f, z_i_f_aug = self.convBlockEncoder_f(X_data_f.float(), X_aug_f.float())  # 频域嵌入
                loss_f = self.loss_f(z_i_f, z_i_f_aug)

                # # 输入原始数据和增强后的数据
                # z_i_t = self.lstmEncoder_t.forward(X)  # torch.Size([4, 512])
                # z_i_t = z_i_t.unsqueeze(0).permute(0, 2, 1).expand(3, 512, 4)
                #
                # z_i_t_aug = self.lstmEncoder_t.forward(self.X_aug_t)  # torch.Size([4, 3, 512])
                # z_i_t_aug = z_i_t_aug.permute(1, 2, 0)
                #
                # # 得到频域的嵌入张量
                # z_i_f, z_i_f_aug = self.convBlockEncoder_f.forward(self.X_data_f.to(torch.float32),
                #                                                    self.X_aug_f.to(torch.float32))
                # z_i_f = z_i_f.permute(0, 2, 1)
                # z_i_f_aug = z_i_f_aug.permute(0, 2, 1)
                # z_i_f_aug = self.convBlockEncoder_f.forward(self.X_aug_f)
                print("zif", z_i_f.shape, z_i_f_aug.shape)
                print("zit", z_i_t.shape, z_i_t_aug.shape)
                total_loss = loss + loss_t + loss_f
                total_loss.backward()
                self.optimizer.step()
                # --------------------------------------------------------------
                i += 1
                if i >= self.nb_steps:  # 达到训练次数
                    break
            # self.scheduler.step()
            epochs += 1
        return self.network  # 训练后的模型

    # 输出嵌入
    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths

      
        varying = bool(numpy.isnan(numpy.sum(X))) 

        test = utils.Dataset(X)  
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
      
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()  # 模型测试

        count = 0
      
        with torch.no_grad():  # 禁用autograd引擎的梯度计算的函数
            for batch in test_generator:  # 批量处理数据
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # if self.win_type=='hanning':
                #     batch = hanning_tensor(batch)
               
                features[count * batch_size: (count + 1) * batch_size] = self.network(batch).cpu()  # 输出到CPU上面。
                count += 1
        # 训练网络
        self.network = self.network.train()  # 切换模型到训练模式
        return features  # 返回嵌入

    
    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM. 每个batch中窗口的数量
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)  # 获取输入数据X的形状，包括样本数量、通道数和时间序列长度
        num_window = int((length - win_size) / step) + 1  # 计算滑动窗口的数量     窗口数量 = (序列长度 - 窗口大小) / 步长 + 1
        embeddings = numpy.empty(
            (num_batch, self.out_channels, num_window))  # 创建一个空的数组，用于存储每个样本的嵌入表示。数组的形状为(样本数量, 输出通道数, 滑动窗口的数量)

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):  # 注意 num_window> window_batch_size
                # 构成一个由三元组X组成的 list, 其中list中的每个元素 相当于一个滑动窗口，步长为step. 窗口大小为j+win_size

                masking = numpy.array([X[b, :, j: j + win_size] for j in range(step * i * window_batch_size,
                                                                               step * min((i + 1) * window_batch_size,
                                                                                          num_window),
                                                                               step)])  # masking.shape = (window_batch_size, num_channel, win_size)
                # print(masking[1][0][1])
                
                if self.win_type == 'hanning':
                    masking = hanning_numpy(masking)  # return weight*X
                print("test", masking.shape, step * i * window_batch_size,
                      step * min((i + 1) * window_batch_size, num_window))
                # 存储到embeddings的特定片段中
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
                    self.encode(masking[:], batch_size=batch_size), 0,
                    1)  # embeddings.shape = (num_batch, out_channels, num_window)   交换了embeddings数组的第0维和第1维
        #
        return embeddings[0].T  # embeddings.shape = (num_batch, out_channels, num_window)

    def set_params(self, compared_length, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size,
            nb_steps, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self


