import numpy
import sys
import os

import sklearn

sys.path.append(os.path.dirname(__file__))
import losses
import networks
import numpy as np
from S2CSD.augmentations import DataTransform_T, DataTransform_F



#sys.path.append(os.path.dirname(__file__))
import utils
import math
import torch
from .losses.TFC_loss import TFCLoss
from .losses.fncc_loss import fncc_loss
from .losses.LSE_loss import LSELoss
# from losses.TFC_loss import TimeDomainLoss_t
# from losses.TFC_loss import FrequencyDomainLoss_f
from .networks.CausalCnn import CausalCNNEncoder
from .networks.ConvEncoder_f import ConvBlockEncoder_f
from .networks.LSTMEncoder_t import LSTMEncoder


"""
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
        pass

    def save(self, X):
        pass

    def load(self, X):
        pass

class CausalConv(BasicEncoderClass):  #
    def __init__(self, win_size, batch_size, nb_steps, lr,
                 channels, depth, reduced_size, out_channels, kernel_size,
                 in_channels, cuda, gpu, M, N, win_type):

       
        self.network = self.__create_network(in_channels, channels, depth,
                                             reduced_size, out_channels, kernel_size, cuda, gpu)

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
        # ---------------------------------------------------------

        self.jitter_scale_ratio = 1.1  
        self.jitter_ratio = 0.8  
        self.max_seg = 2  # 


        self.remove_frequency_ratio = 0.1  
        self.add_frequency_ratio = 0.1  

        # 时域LSTM参数
        # self.feature_dim = 256 
        self.hidden_size = 1024 
        self.output_size = 512  

        self.dropout_rate = 0.10  
        self.num_layers = 2  

      
        #self.input_channels = 4  
        self.kernel_size = 4  
        self.stride = 1  # 
        self.output_channels = 4  # 
        # self.num_classes = None  # 
        self.dropout = 0.30  #
        self.batchsize = 1  #

       
        self.loss = losses.TFC_loss.TFCLoss(
            win_size, M, N, win_type
        )  # 

      
        self.loss_t =losses.TFC_loss.TimeDomainLoss_t(device="cuda")
      
        self.loss_f =losses.TFC_loss.FrequencyDomainLoss_ftest(device="cuda")


      
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)  
        # ---------------------------------------------------------------
        self.loss_list = [] 

 
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        # -------------------------------------------------

      
        network = networks.CausalCnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double() 
        if cuda:
            network.cuda(gpu)

        return network

  
    def fit(self, X, save_memory=False, verbose=False):

        self.input_channels = X.shape[1]
        train = torch.from_numpy(X) 
        if self.cuda:
            train = train.cuda(self.gpu)  

        # -------------------------------------------------------

      
        train_torch_dataset = utils.Dataset(X) 
      
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        ) 

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            # ---------------------------------------
            for batch in train_generator:  
                if self.cuda:
                    batch = batch.cuda(self.gpu) 
                # --------------------------------
               
                self.optimizer.zero_grad()
             
                # losses.LSE_loss.LSELoss()
                # print("batch:",batch.shape)
                # print("XX:",X.shape)

             
                loss = self.loss(batch, self.network, save_memory=save_memory)

                self.lstmEncoder_t = networks.LSTMEncoder_t.LSTMEncoder(batch.shape[2], self.hidden_size,
                                                                        self.num_layers, self.output_size,
                                                                        self.dropout_rate)

                self.convBlockEncoder_f = networks.ConvEncoder_f.ConvBlockEncoder_f(self.input_channels,
                                                                                    self.output_channels,
                                                                                    self.kernel_size)
               #------------


                self.x_data_f = torch.fft.fft(torch.from_numpy(X)).abs()  # torch.Size([1, 4, 5673])


                self.X_aug_t = DataTransform_T(X, self.jitter_ratio, self.jitter_scale_ratio, self.max_seg)

                #-----------
                self.X_data_f, self.X_aug_f = DataTransform_F(self.x_data_f, self.remove_frequency_ratio, self.add_frequency_ratio)

             
                z_i_t = self.lstmEncoder_t.forward(X)  # torch.Size([4, 512])
                # ========================================
             
                # z_i_t =z_i_t.unsqueeze(0).unsqueeze(0)
                # z_i_t = z_i_t.expand(1,3,z_i_t.shape[2])
                #===========================================
                #print("zzzz,z_i_t",z_i_t.shape)
            

                z_i_t = z_i_t.unsqueeze(0).expand(3,z_i_t.shape[0],z_i_t.shape[1])


                z_i_t_aug = self.lstmEncoder_t.forward(self.X_aug_t)  # torch.Size([3, 4, 512])
                #=================================

              
                z_i_t_aug = z_i_t_aug.unsqueeze(0)
                #==========================================
                #z_i_t_aug = z_i_t_aug.permute(0, 2, 1)
                #z_i_t_aug = z_i_t_aug.unsqueeze(0)  #exp_on_UCR_SEG
                #------------------------------------------------------
           
                z_i_f, z_i_f_aug = self.convBlockEncoder_f.forward(self.X_data_f.to(torch.float32),
                                                                   self.X_aug_f.to(torch.float32))
                z_i_f = z_i_f.permute(0, 2, 1)
                z_i_f_aug = z_i_f_aug.permute(0, 2, 1)
                # z_i_f_aug = self.convBlockEncoder_f.forward(self.X_aug_f)
                #print("zif", z_i_f.shape, z_i_f_aug.shape)
                #print("zit", z_i_t.shape, z_i_t_aug.shape)
               
                loss_t = self.loss_t(z_i_t, z_i_t_aug)  

                loss_f = self.loss_f(z_i_f, z_i_f_aug)
                # print("loss_t_f", loss_t,loss_f)
                weight1 = 0.6
                weight2 = 0.2
                weight3 = 0.2
                total_loss =  weight1*loss + weight2*loss_t + weight3*loss_f
                #total_loss = weight1 * loss + weight3 * loss_f #notime
                #total_loss =  weight1*loss + weight2*loss_t #nofrequency

               # print("total_loss", total_loss,loss, loss_f, loss_t)

                total_loss.backward()  
                self.optimizer.step()  
                # --------------------------------------------------------------
                i += 1
                if i >= self.nb_steps:  
                    break
            # self.scheduler.step()
            epochs += 1
        return self.network  

 
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

        self.network = self.network.eval()  

        count = 0
      
        with torch.no_grad(): 
            for batch in test_generator:  
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # if self.win_type=='hanning':
                #     batch = hanning_tensor(batch)
            
                features[count * batch_size: (count + 1) * batch_size] = self.network(batch).cpu()  
                count += 1
       
        self.network = self.network.train() 
        return features  

 
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
               run of encode, to save RAM. 
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)  
        num_window = int((length - win_size) / step) + 1  
        embeddings = numpy.empty(
            (num_batch, self.out_channels, num_window))  

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):  # 注意 num_window> window_batch_size
                # 构成一个由三元组X组成的 list, 其中list中的每个元素 相当于一个滑动窗口，步长为step. 窗口大小为j+win_size
                # 窗口数据
                masking = numpy.array([X[b, :, j: j + win_size] for j in range(step * i * window_batch_size,
                                                                               step * min((i + 1) * window_batch_size,
                                                                                          num_window),
                                                                               step)])  # masking.shape = (window_batch_size, num_channel, win_size)
                # print(masking[1][0][1])
                # 但是这有什么区别？
                if self.win_type == 'hanning':
                    masking = hanning_numpy(masking)  # return weight*X
                # print("test", masking.shape, step * i * window_batch_size,
                #       step * min((i + 1) * window_batch_size, num_window))
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


#===============================================================================
"""
上面的部分和 Time2station 部分相同
"""
#========================================================================
# 这部分构建E2USD的编码器 完成对比损失函数 fncc_Loss的实验

class CausalConv_fnccLoss(BasicEncoderClass):
    def __init__(self, win_size, batch_size, nb_steps, lr,
                 channels, depth, reduced_size, out_channels, kernel_size,
                 in_channels, cuda, gpu, M, N, win_type):

        # 构造网络结构，类似 time2station
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                             out_channels, kernel_size, cuda, gpu)
        self.win_type = win_type
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        #==================================================================
        # E2usd 改变了 loss，这部分与 time2station 不同，但参数配置一样
        # networks/fncc_los 返回 loss 用于消去 false negtive pair 的影响
        self.loss = losses.fncc_loss.fncc_loss(
            win_size, M, N, win_type
        )
        # 网络参数更新，保存需要更新的参数
        params_to_update = [p for p in self.network.parameters() if p.requires_grad]
        #============================================================================
        self.optimizer = torch.optim.Adam(params_to_update, lr=lr)

        self.loss_list = []

    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        """
        创建双端嵌入网络。

        :param in_channels: 输入通道数
        :param channels: 通道数
        :param depth: 网络深度
        :param reduced_size: 降维后的大小
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param cuda: 是否使用 CUDA
        :param gpu: 使用的 GPU 编号
        :return: 创建好的网络模型
        """
        # 双端嵌入模块，在 networks/network 下面
        # 返回 embedding, trend_x_embedding, seasonal_x_embedding
        #=================================================================
        # 这部分与 time2station 不同，但参数配置一样。使用一个新的网络编码模型来构建网络
        # 增加 double view decomposition  CausalCNNEncoder
        network = networks.CausalCnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        #======================================================================
        # 将网络参数存储为 double 型
        network.double()
        if cuda:
            network.cuda(gpu)
        return network  # 返回嵌入网络

    #============================================================
    # 和之前代码相同
    def fit(self, X, save_memory=False, verbose=False):
        """
        训练网络模型。

        :param X: 输入的训练数据
        :param save_memory: 是否节省内存，默认为 False
        :param verbose: 是否打印详细信息，默认为 False
        :return: 训练好的网络模型
        """
        # 将输入数据转换为 torch.Tensor
        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        # 调用 utils 中的 Dataset 类创建数据集
        train_torch_dataset = utils.Dataset(X)
        # 数据加载器，用于批量处理数据
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )
        i = 0

        while i < self.nb_steps:
            # 遍历批量数据
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # 清空优化器的梯度
                self.optimizer.zero_grad()
                # 计算损失函数
                loss = self.loss(batch, self.network, save_memory=False)
                # 反向传播计算梯度
                loss.backward()
                # 更新网络参数
                self.optimizer.step()

                i += 1
                if i >= self.nb_steps:
                    break
        return self.network  # 返回训练好的网络模型

    def encode(self, X, batch_size=500):
        """
        对输入数据进行编码。

        :param X: 输入数据
        :param batch_size: 批量大小，默认为 500
        :return: 编码后的特征
        """
        #print("dadfadf",X.shape)
        # 检查输入数据是否包含 NaN 值
        varying = bool(numpy.isnan(numpy.sum(X)))

        # 创建测试数据集
        test = utils.Dataset(X)
        # 创建测试数据加载器
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )

        # 初始化编码后的特征数组
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        # 将网络设置为评估模式
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # 获取编码后的特征并存储到 features 数组中
                features[ count * batch_size: (count + 1) * batch_size] = self.network(batch)[0].cpu()
                count += 1

        return features

    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        """
        对输入数据按窗口进行编码。

        :param X: 输入数据
        :param win_size: 窗口大小，默认为 128
        :param batch_size: 批量大小，默认为 500
        :param window_batch_size: 窗口批量大小，默认为 10000
        :param step: 窗口滑动步长，默认为 10
        :return: 窗口编码后的嵌入向量
        """
        num_batch, num_channel, length = numpy.shape(X)
        # 计算窗口数量
        num_window = int((length - win_size) / step) + 1
        # 初始化嵌入向量数组
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            # 计算批量的次数
            for i in range(math.ceil(num_window / window_batch_size)):
                # 生成窗口数据
                masking = numpy.array([ X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
                                                                              step * min((i + 1) * window_batch_size,num_window), step)])
                # 对窗口数据进行编码，并交换轴
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
                    self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T

    # 参数设定
    def set_params(self, compared_length, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size,
            nb_steps, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self
#===============================================================
# 增加Time2State
# 构建一个LSE-loss的编码器
class CausalConv_LSE(BasicEncoderClass): # 因果卷积 中使用了LSE损失
    def __init__(self, win_size, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, M, N, win_type):
        # 创建一个网络模型
        self.network = self.__create_network(in_channels, channels, depth,
                                    reduced_size,out_channels, kernel_size, cuda, gpu)

        self.win_type = win_type
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        #----------------------------------------------------------
        # 损失函数  这是LSE_loss.py中 实现的LSE-loss函数
        self.loss = losses.LSE_loss.LSELoss(
            win_size, M, N, win_type
        ) # 损失函数
        #----------------------------------------------------------------
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr) # Adam优化器
        #---------------------------------------------------------------
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.98, -1)
        self.loss_list = []  # 损失函数列表

    # 创建网络模型, 创建因果卷积网络
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        #-------------------------------------------------
        # 使用因果卷积编码器创建了一个网络模型
        network = networks.CausalCnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()  #  将数据类型设置为双精度浮点数
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
            train = train.cuda(self.gpu) #使用GPU

        #-------------------------------------------------------

        #训练数据集
        # 创建一个PyTorch数据集对象，用于加载训练数据集
        train_torch_dataset = utils.Dataset(X)   # PyTorch wrapper for a numpy dataset. 是一个numpy dataset对象
        # 用于创建数据加载器（data loader）的函数。数据加载器用于加载训练和测试数据集，并将数据划分为小批量进行处理
        # 可以方便地进行数据批量处理、乱序加载和并行读取
        #
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        ) # 加载训练数据集

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            #---------------------------------------
            for batch in train_generator:  #train_generator中的批量数据
                if self.cuda:
                    batch = batch.cuda(self.gpu) # 使用GPU
                #--------------------------------
                # 下面部分是训练模型
                self.optimizer.zero_grad()
                # 损失函数 预测和真实标签之间的损失
                # losses.LSE_loss.LSELoss()
                loss = self.loss(batch, self.network, save_memory=save_memory)
                loss.backward()           # 反向传播计算梯度
                self.optimizer.step() # 优化模型
                #--------------------------------------------------------------
                i += 1
                if i >= self.nb_steps:  #达到训练次数
                    break
            # self.scheduler.step()
            epochs += 1
        return self.network  #训练后的模型

   # 输出嵌入
    def encode(self, X, batch_size=500):

       # 判断数据中是否有nan 非数值""
        varying = bool(numpy.isnan(numpy.sum(X))) # 主要判断x中是否存在非数值, np.isnan是全false, bool 也是false,说明数据中没有非数值值

        test = utils.Dataset(X) #加载的数据集
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        # 构造矩阵  batch X out_channels
        # 获取输入数据X的样本数量（第一个维度）
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval() # 模型测试

        count = 0
        # no_grad()方法是用于在评估模型性能时禁用autograd引擎的梯度计算的函数
        with torch.no_grad(): # 禁用autograd引擎的梯度计算的函数
            for batch in test_generator:# 批量处理数据
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # if self.win_type=='hanning':
                #     batch = hanning_tensor(batch)
                # 将结果输出到CPU上面。保存在features中
                features[count * batch_size : (count + 1) * batch_size] = self.network(batch).cpu() # 输出到CPU上面。
                count += 1
        #训练网络
        self.network = self.network.train() # 切换模型到训练模式
        return features # 返回嵌入

     # 网络窗口编码器
    # 对于给定大小的输入的每个子序列(滑动窗口表示)，输出编码器与输入相关联的表示。
    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):

        num_batch, num_channel, length = numpy.shape(X) # 获取输入数据X的形状，包括样本数量、通道数和时间序列长度
        num_window = int((length-win_size)/step)+1 # 计算滑动窗口的数量     窗口数量 = (序列长度 - 窗口大小) / 步长 + 1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))  # 创建一个空的数组，用于存储每个样本的嵌入表示。数组的形状为(样本数量, 输出通道数, 滑动窗口的数量)
        for b in range(num_batch):
            for i in range(math.ceil(num_window/window_batch_size)): #注意 num_window> window_batch_size
                #构成一个由三元组X组成的 list, 其中list中的每个元素 相当于一个滑动窗口，步长为step. 窗口大小为j+win_size

                masking = numpy.array( [ X[b ,: , j : j+win_size] for j in range( step*i*window_batch_size,
                                step * min( (i+1)* window_batch_size, num_window ), step ) ] )  # masking.shape = (window_batch_size, num_channel, win_size)

                if self.win_type=='hanning':
                    masking = hanning_numpy(masking) # return weight*X
                print("test",masking.shape, step*i*window_batch_size, step * min( (i+1)* window_batch_size, num_window) )
                # 存储到embeddings的特定片段中
                embeddings[b,:,i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(self.encode( masking[:], batch_size=batch_size ), 0, 1) # embeddings.shape = (num_batch, out_channels, num_window)   交换了embeddings数组的第0维和第1维
       #
        return embeddings[0].T # embeddings.shape = (num_batch, out_channels, num_window)

    def set_params(self, compared_length, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size,
            nb_steps, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self
#---------------------------------------------------------------

class LSTM_LSE(BasicEncoderClass):  # LSTM编码器
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 channels, depth, reduced_size, out_channels, kernel_size,
                 in_channels, cuda, gpu, M, N):
        # 网络模型
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                             out_channels, kernel_size, cuda, gpu)
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 损失函数LSE_loss(self)
        self.loss = losses.LSE_loss.LSELoss(
            compared_length, nb_random_samples, negative_penalty, M, N
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adagrad(self.network.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=lr)
        self.loss_list = []

    # 创建模型
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        # RNN的编码器 可以选择GRU和LSTM (这个地方和上面的地方存在区别 使用LSTM或者GRU的方法)
        network = networks.rnn.RnnEncoder(256, in_channels, out_channels, num_layers=2, cell_type='GRU', device='cuda',
                                          dropout=0.1)
        # network.double()
        if cuda:
            network.cuda(gpu)
        return network

    # 使用给定的训练数据,无监督地训练编码器。
    def fit(self, X, y=None, save_memory=False, verbose=False):


        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            # unique是返回数组中的唯一数，
            # 类别的数量
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]  # 训练数据集大小
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0  #
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        while i < self.nb_steps:  # 执行优化的步骤数小于总的步数
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                # print(batch.size(2))
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                # 主要实现了早停策略（Early stopping strategy）的部分逻辑
                if not varying:
                    # 损失
                    loss = self.loss(
                        batch, self.network, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.network, train, save_memory=save_memory
                    )
                # 损失分离 单独保存一个损失函数
                self.loss_list.append(loss.detach().cpu().numpy())
                loss.backward()  # 后向传播

                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:  # 如果迭代地步数大于总的步数
                    break
            epochs += 1
            # Early stopping strategy 早弃法
            if self.early_stopping is not None and y is not None and (
                    ratio >= 5 and train_size >= 50  # 调整学习率和训练大小
            ):
                # Computes the best regularization parameters
                features = self.encode(X)  # 编码器 返回features，相当于提取了x中特征
                # 基于提取的特征训练分类器
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score  实现交叉验证  通过交叉验证来评估分数
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:  # 每个要更新最大得分,找到一个最好的得分项
                    count = 0
                    found_best = True
                    max_score = score  # 更新模型的得分
                    best_encoder = type(self.network)(**self.params)  # #？？？？
                    best_encoder.double()  #
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    # 装载状态模型
                    best_encoder.load_state_dict(self.network.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.network

    # 编码
    def encode(self, X, batch_size=5000):
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
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # 特征数据
                    features[count * batch_size: (count + 1) * batch_size] = self.network(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # 计算除去 存在非数值个数以外 序列的长度
                    length = batch.size(2) - torch.sum(torch.isnan(batch[0, 0])).data.cpu().numpy()
                    features[count: count + 1] = self.network(batch[:, :, :length]).cpu()
                    count += 1
        self.network = self.network.train()
        return features

    # 编码窗口序列
    def encode_window(self, X, win_size=128, batch_size=50, window_batch_size=1000, step=10):
        """
        Encode a time series.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features).

        win_size : even integer.
            Size of window.

        batch_size : integer.
            Batch size when encoding.

        window_batch_size : integer.

        step : integer.
            Step size of sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)  #
        num_window = int((length - win_size) / step) + 1
        # 构造一个空矩阵
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))
        # 按照批量
        for b in range(num_batch):
            # window_batch_size
            for i in range(math.ceil(num_window / window_batch_size)):  # window_batch_size是每个窗口中batch_size的大小
                masking = numpy.array([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
                                                                              step * min((i + 1) * window_batch_size,
                                                                                         num_window), step)])
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
                    self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self  # 返回对象

#
# class CausalConv_f():
#     def __init__(self, win_size, batch_size, nb_steps, lr,
#                  channels, strike, reduced_size, out_channels, kernel_size,
#                  in_channels, cuda, gpu, M, N, win_type,temprature):
#        
#
#         self.network = self.__create_network(in_channels, channels, strike, reduced_size,
#                                              out_channels, kernel_size, cuda, gpu)
#
#         self.win_type = win_type
#         self.architecture = ''
#         self.cuda = cuda
#         self.gpu = gpu
#         self.batch_size = batch_size
#         self.nb_steps = nb_steps
#         self.lr = lr
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.strike  = strike
#
#         # ==================================================================
#         # 这个部分使用了频域上的loss
#         self.loss = losses.TFC_loss.TFCLoss(
#             win_size, M, N, win_type
#         )  # 损失函数
#
#         # 网络参数更新，保存需要更新的参数
#         params_to_update = [p for p in self.network.parameters() if p.requires_grad]
#         # ============================================================================
#         self.optimizer = torch.optim.Adam(params_to_update, lr=lr)
#
#         self.loss_list = []
#
#     def __create_network(self, in_channels, channels, strike, padding,
#                          out_channels, kernel_size, cuda, gpu):
#        # 这里需要卷积训练。 三个卷积块
#         network = networks.ConvEncoder_f.ConvBlockEncoder_f(
#             in_channels, channels, strike, padding, out_channels,
#             kernel_size
#         )
#         # ======================================================================
#         # 将网络参数存储为 double 型
#         network.double()
#         if cuda:
#             network.cuda(gpu)
#
#         return network  # 返回嵌入网络
#
#         # ============================================================
#         # 和之前代码相同
#
#     def fit(self, X, save_memory=False, verbose=False):
#         """
#         训练网络模型。
#
#         :param X: 输入的训练数据
#         :param save_memory: 是否节省内存，默认为 False
#         :param verbose: 是否打印详细信息，默认为 False
#         :return: 训练好的网络模型
#         """
#         # 将输入数据转换为 torch.Tensor
#         train = torch.from_numpy(X)
#         if self.cuda:
#             train = train.cuda(self.gpu)
#
#         # 调用 utils 中的 Dataset 类创建数据集
#         train_torch_dataset = utils.Dataset(X)
#         # 数据加载器，用于批量处理数据
#         train_generator = torch.utils.data.DataLoader(
#             train_torch_dataset, batch_size=self.batch_size, shuffle=True
#         )
#         i = 0
#
#         while i < self.nb_steps:
#             # 遍历批量数据
#             for batch in train_generator:
#                 if self.cuda:
#                     batch = batch.cuda(self.gpu)
#                 # 清空优化器的梯度
#                 self.optimizer.zero_grad()
#                 # 计算损失函数
#                 # 这里采用一个频域上的损失函数
#                 loss = self.loss_f(batch, self.network, save_memory=False)
#                 # 反向传播计算梯度
#                 loss.backward()
#                 # 更新网络参数
#                 self.optimizer.step()
#
#                 i += 1
#                 if i >= self.nb_steps:
#                     break
#
#         return self.network  # 返回训练好的网络模型
#
#     def encode(self, X, batch_size=500):
#         """
#         对输入数据进行编码。
#
#         :param X: 输入数据
#         :param batch_size: 批量大小，默认为 500
#         :return: 编码后的特征
#         """
#
#         # 检查输入数据是否包含 NaN 值
#         varying = bool(numpy.isnan(numpy.sum(X)))
#
#         # 创建测试数据集
#         test = utils.Dataset(X)
#         # 创建测试数据加载器
#         test_generator = torch.utils.data.DataLoader(
#             test, batch_size=batch_size if not varying else 1
#         )
#
#         # 初始化编码后的特征数组
#         features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
#         # 将网络设置为评估模式
#         self.network = self.network.eval()
#
#         count = 0
#         with torch.no_grad():
#             for batch in test_generator:
#                 if self.cuda:
#                     batch = batch.cuda(self.gpu)
#                 # 获取编码后的特征并存储到 features 数组中
#                 features[count * batch_size: (count + 1) * batch_size] = self.network(batch)[0].cpu()
#                 count += 1
#
#         return features
#
#     def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
#         """
#         对输入数据按窗口进行编码。
#
#         :param X: 输入数据
#         :param win_size: 窗口大小，默认为 128
#         :param batch_size: 批量大小，默认为 500
#         :param window_batch_size: 窗口批量大小，默认为 10000
#         :param step: 窗口滑动步长，默认为 10
#         :return: 窗口编码后的嵌入向量
#         """
#         num_batch, num_channel, length = numpy.shape(X)
#         # 计算窗口数量
#         num_window = int((length - win_size) / step) + 1
#         # 初始化嵌入向量数组
#         embeddings = numpy.empty((num_batch, self.out_channels, num_window))
#
#         for b in range(num_batch):
#             # 计算批量的次数
#             for i in range(math.ceil(num_window / window_batch_size)):
#                 # 生成窗口数据
#                 masking = numpy.array([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
#                                                                               step * min((i + 1) * window_batch_size,
#                                                                                          num_window), step)])
#                 # 对窗口数据进行编码，并交换轴
#                 embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
#                     self.encode(masking[:], batch_size=batch_size), 0, 1)
#
#         return embeddings[0].T
#
#         # 参数设定
#
#     def set_params(self, compared_length, batch_size, nb_steps, lr,
#                    channels, depth, reduced_size, out_channels, kernel_size,
#                    in_channels, cuda, gpu):
#
#         self.__init__(
#             compared_length, batch_size,
#             nb_steps, lr, channels, depth,
#             reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
#         )
#         return self
#
# # 增加一个时域编码器 LSTM
# class LSTMEncoder_t():
#     def __init__(self, win_size, batch_size, nb_steps, lr,
#                  channels, strike, reduced_size, out_channels, kernel_size,
#                  in_channels, cuda, gpu, M, N, win_type,temprature):
#         # 创建一个频域上的网络结构
#         # 这个网络结构是三个卷积块 在ConvEncoder_f.py中
#
#         self.network = self.__create_network(in_channels, channels, strike, reduced_size,
#                                              out_channels, kernel_size, cuda, gpu)
#
#         self.win_type = win_type
#         self.architecture = ''
#         self.cuda = cuda
#         self.gpu = gpu
#         self.batch_size = batch_size
#         self.nb_steps = nb_steps
#         self.lr = lr
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.strike  = strike
#
#         # ==================================================================
#         # 这个部分使用了频域上的loss
#         self.loss_f =losses.TFC_loss.FrequencyDomainLoss_f(
#            batch_size,temprature
#         )
#
#         # 网络参数更新，保存需要更新的参数
#         params_to_update = [p for p in self.network.parameters() if p.requires_grad]
#         # ============================================================================
#         self.optimizer = torch.optim.Adam(params_to_update, lr=lr)
#
#         self.loss_list = []
#
#
#
#     def __create_network(self, in_channels, channels, strike, padding,
#                          out_channels, kernel_size, cuda, gpu):
#        # 这里需要卷积训练。 三个卷积块
#         network = networks.ConvEncoder_f.ConvBlockEncoder_f(
#             in_channels, channels, strike, padding, out_channels,
#             kernel_size
#         )
#         # ======================================================================
#         # 将网络参数存储为 double 型
#         network.double()
#         if cuda:
#             network.cuda(gpu)
#
#         return network  # 返回嵌入网络
#
#         # ============================================================
#         # 和之前代码相同
#
#     def fit(self, X, save_memory=False, verbose=False):
#         """
#         训练网络模型。
#
#         :param X: 输入的训练数据
#         :param save_memory: 是否节省内存，默认为 False
#         :param verbose: 是否打印详细信息，默认为 False
#         :return: 训练好的网络模型
#         """
#         # 将输入数据转换为 torch.Tensor
#         train = torch.from_numpy(X)
#         if self.cuda:
#             train = train.cuda(self.gpu)
#
#         # 调用 utils 中的 Dataset 类创建数据集
#         train_torch_dataset = utils.Dataset(X)
#         # 数据加载器，用于批量处理数据
#         train_generator = torch.utils.data.DataLoader(
#             train_torch_dataset, batch_size=self.batch_size, shuffle=True
#         )
#         i = 0
#
#         while i < self.nb_steps:
#             # 遍历批量数据
#             for batch in train_generator:
#                 if self.cuda:
#                     batch = batch.cuda(self.gpu)
#                 # 清空优化器的梯度
#                 self.optimizer.zero_grad()
#                 # 计算损失函数
#                 # 这里采用一个频域上的损失函数
#                 loss = self.loss_f(batch, self.network, save_memory=False)
#                 # 反向传播计算梯度
#                 loss.backward()
#                 # 更新网络参数
#                 self.optimizer.step()
#
#                 i += 1
#                 if i >= self.nb_steps:
#                     break
#
#         return self.network  # 返回训练好的网络模型
#
#     def encode(self, X, batch_size=500):
#         """
#         对输入数据进行编码。
#
#         :param X: 输入数据
#         :param batch_size: 批量大小，默认为 500
#         :return: 编码后的特征
#         """
#
#         # 检查输入数据是否包含 NaN 值
#         varying = bool(numpy.isnan(numpy.sum(X)))
#
#         # 创建测试数据集
#         test = utils.Dataset(X)
#         # 创建测试数据加载器
#         test_generator = torch.utils.data.DataLoader(
#             test, batch_size=batch_size if not varying else 1
#         )
#
#         # 初始化编码后的特征数组
#         features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
#         # 将网络设置为评估模式
#         self.network = self.network.eval()
#
#         count = 0
#         with torch.no_grad():
#             for batch in test_generator:
#                 if self.cuda:
#                     batch = batch.cuda(self.gpu)
#                 # 获取编码后的特征并存储到 features 数组中
#                 features[count * batch_size: (count + 1) * batch_size] = self.network(batch)[0].cpu()
#                 count += 1
#
#         return features
#
#     def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
#         """
#         对输入数据按窗口进行编码。
#
#         :param X: 输入数据
#         :param win_size: 窗口大小，默认为 128
#         :param batch_size: 批量大小，默认为 500
#         :param window_batch_size: 窗口批量大小，默认为 10000
#         :param step: 窗口滑动步长，默认为 10
#         :return: 窗口编码后的嵌入向量
#         """
#         num_batch, num_channel, length = numpy.shape(X)
#         # 计算窗口数量
#         num_window = int((length - win_size) / step) + 1
#         # 初始化嵌入向量数组
#         embeddings = numpy.empty((num_batch, self.out_channels, num_window))
#
#         for b in range(num_batch):
#             # 计算批量的次数
#             for i in range(math.ceil(num_window / window_batch_size)):
#                 # 生成窗口数据
#                 masking = numpy.array([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
#                                                                               step * min((i + 1) * window_batch_size,
#                                                                                          num_window), step)])
#                 # 对窗口数据进行编码，并交换轴
#                 embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
#                     self.encode(masking[:], batch_size=batch_size), 0, 1)
#
#         return embeddings[0].T
#
#         # 参数设定
#
#     def set_params(self, compared_length, batch_size, nb_steps, lr,
#                    channels, depth, reduced_size, out_channels, kernel_size,
#                    in_channels, cuda, gpu):
#
#         self.__init__(
#             compared_length, batch_size,
#             nb_steps, lr, channels, depth,
#             reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
#         )
#         return self

