import torch
from torch import nn
import math

from torchvision import transforms
from torchvision.transforms.transforms import ToTensor


class HandLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_size = input_sz  # 输入维度
        self.hidden_size = hidden_sz # 隐藏层节点数
        
        # 可以使用nn.Parameter()来转换一个固定的权重数值，使的其可以跟着网络训练一直调优下去，学习到一个最适合的权重值。
        # 作用是使得当前的参数可以被保存梯度
        # 类似于 y = wx + b 的格式  ，这里为了方便计算将w分为了w和u 分别作为x和h的权重
        #f_t  遗忘门，第一个sigmoid 函数
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))  # 28*128
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz)) # 128*128
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))            # 128

        #i_t 输入门，第二个sigmoid函数
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))   # 28*128
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))  # 128*128
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))             # 128
        

        #o_t  输出门 ， 第三个 sigmoid 函数
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz)) # 28*128
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz)) # 128*128
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))            # 128
      
        #c_t  长期记忆
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))  # 28*128
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz)) # 128*128
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))            # 128
        
       
        
        self.init_weights()  # 初始化权重

    
    def init_weights(self):
        """
        权重初始化方法，将其用作PyTorch默认值中的权重初始化
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)   # uniform_(a,b)  在a,b中间tensor从均匀分布中抽样数值进行填充。
    
    def forward(self,x,init_states=None):
        
        """
        前向计算
        x.size表示（批次大小、序列大小、输入大小）
       
        """
        bs, seq_sz, _ = x.size() # 32,28,28  
        hidden_seq = []
        
        if init_states is None:  # 如果初始化的声明为空
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),  # t0 时刻前的h_t初始状态
                torch.zeros(bs, self.hidden_size).to(x.device),  # t0 时刻前的c_t初始状态
            )
        else:
            h_t, c_t = init_states
            
        for t in range(seq_sz):  # 28  也就是这个图片是一个 28*28的，每次传入一行就是 28 个数值，然后循环28次
            x_t = x[:, t, :]
            
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)    #  第一个激活函数  遗忘门计算
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)     # 第二个激活函数
            g_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)        # 第一个双曲正切  
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)     # 第三个激活函数   

 
            c_t = f_t * c_t + i_t * g_t   # 长期记忆计算   遗忘门（ft）*上一个时刻的长期记忆（ct）  + 输入门（it） *  经过双曲正切得到的c't
            h_t = o_t * torch.tanh(c_t)    # 结果计算 
            
            hidden_seq.append(h_t.unsqueeze(0))  # unsqueeze(0)  在第 0 维增加一个维度
        
        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)  # cat(0)  按第0维进行拼接  将这个list 转化为tensor结构
        hidden_seq = hidden_seq.transpose(0, 1)   # transpose(0, 1)  转置 


        return  hidden_seq, (h_t, c_t)


class Net(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_class,bi=False):
        super().__init__()
        self.bi =bi
        self.lstm = HandLSTM(input_sz,hidden_sz) #nn.LSTM(input_sz,hidden_sz, batch_first=True)
        if bi == True:                              # 判断是否双向
            hidden_sz = hidden_sz*2
        self.fc1 = nn.Linear(hidden_sz, num_class)   #  线性
        
    def forward(self, x):
        if self.bi == False:        
            x_, (h_n, c_n) = self.lstm(x)  
            x_ = (x_[:, -1, :])
            x_ = self.fc1(x_)
        else:
            x_, (h_n, c_n) = self.lstm(x)  
            x_ = (x_[:, -1, :])
            x_f = x.flip(0)
            x_f, (h_n, c_n) = self.lstm(x_f)  
            x_f = (x_f[:, -1, :])
            x_bi = torch.hstack([x_,x_f])
            x_ = self.fc1(x_bi)
          
           
        return x_

class LSTM_Classfication(nn.Module):
    def __init__(self, input_size, hidden_size, num_class,bi=False):
        super(LSTM_Classfication, self).__init__()
 
        self.rnn = nn.LSTM(         # LSTM模型
            input_size = input_size,      # 图片每行的数据像素点(特征数)
            hidden_size = hidden_size,     # rnn 隐藏层单元数
            num_layers = 1,       # 层数
            batch_first = True,   # 指定batch为第一维 e.g. (batch, time_step, input_size)
            bidirectional = bi 
        )
        if bi == True:
            hidden_size = hidden_size*2
        self.out = nn.Linear(hidden_size, num_class)    # 输出层
 
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) rnn hidden
        r_out, (c_n,h_n) = self.rnn(x)   
        
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:,-1,:])
        # out = self.out(h_n)
        return out