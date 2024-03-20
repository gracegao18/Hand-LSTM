import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
from hand_lstm import *
import pickle
import os 

# 模型保存路径
path = os.path.join(os.path.dirname(__file__),'model','lstm_model.mod')
# 检测可用设备
device = "cuda" if torch.cuda.is_available() else "cpu"   # 设备
print(device)

def train(train_loader, test_loader, model):
    """
    训练方法，输入训练集，测试集和模型
    """
    optimizer = torch.optim.Adam(lstm.parameters(), lr = lr)   # 优化器，优化所有参数   Adam 优化算法是一种对随机梯度下降法的扩展   lr = 学习率
    loss_func = nn.CrossEntropyLoss()   # 交叉熵损失函数
    size = len(train_loader.dataset)   # 取出训练集的长度
    for _ in range(epochs):   # 按轮次进行循环
        model.train()        # 启动训练模式 
        for step, (x, y) in enumerate(train_loader):   

            # .to(device) 放到显存里
            b_x = Variable(x.reshape(-1, 28, 28).to(device))   # reshape x to (batch, time_step, input_size)
            b_y = Variable(y.to(device))               # batch y   

            output = model(b_x)             # rnn 输出
            loss = loss_func(output, b_y)   # 交叉熵损失
            optimizer.zero_grad()           # 清理上一次更新的梯度值
            loss.backward()                 # 后向传播，计算梯度
            optimizer.step()                # 更新梯度值

            if step % 100 == 0:
                loss, current = loss.item(), step * len(b_x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # 每个epochs训练后测试
        test(test_loader, model, loss_func)

def test(dataloader, model, loss_fn):
    """
    测试方法，传入测试数据，模型和损失函数
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 模型设置为评估模式，代码等效于 model.train(False)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)       # 同上 放在显存里
            pred = model(X.reshape(-1, 28, 28))     # 获得预测值
            test_loss += loss_fn(pred, y).item()    # 求取损失值的和
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  
    test_loss /= num_batches           
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 保存模型

def save_model(model, pickle_file):
    # 模型保存
    with open(pickle_file,'wb') as f:
        pickle.dump(model,f)
        print('模型保存成功!')

def load_model(pickle_file):
   # 模型读取
    with open(pickle_file,'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    # 模型训练超参数
    epochs = 1           # 训练几轮
    batch_size = 32      # 每次训练样本批次大小
    hidden_size = 128    # rnn 模型隐藏层参数大小
    time_step = 28      # rnn 时间步数 / 图片高度
    input_size = 28     # rnn 每步输入值 / 图片每行像素
    num_class = 10      # 预测的类别数
    lr = 0.001           # 学习率

    # Mnist 手写数字
    train_data = dsets.MNIST(root = 'data_sets/mnist', #选择数据的根目录
                            train = True, # 选择训练集
                            transform = transforms.ToTensor(), #转换成tensor变量
                            download = True) # 从网络上download图片
 
    test_data = dsets.MNIST(root = 'data_sets/mnist', #选择数据的根目录
                            train = False, # 选择测试集
                            transform = transforms.ToTensor(), #转换成tensor变量
                            download = True) # 从网络上download图片
 
    # 批训练 50samples, 28x28 (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data)


    # 创建lstm推理模型
    if os.path.exists(path):
        lstm = load_model(path)
        print('模型加载成功')
    else:
        # lstm = Net(input_size, hidden_size, num_class,bi=False) 
        lstm = LSTM_Classfication(input_size, hidden_size, num_class,bi=False) 
        print('模型创建成功')
    lstm.to(device)
    print(lstm)
    
    # training and testing
    train(train_loader, test_loader, lstm)
    # 模型保存
    save_model(lstm, path)
