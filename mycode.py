import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import tensorboard
from torch.utils.tensorboard import SummaryWriter

train_data = pd.read_csv('kaggle_house_pred_train.csv')
test_data = pd.read_csv('kaggle_house_pred_test.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(train_data.shape)
# print(test_data.shape)
#
# print(train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]])

#数据预处理
#因为是竞赛 将训练与测试数据合并 如何对数据进行预处理
all_features = pd.concat((train_data.iloc[:,1:],test_data.iloc[:,1:]))
#标准化数据
numer_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numer_features] = all_features[numer_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
#缺失值填均值0
all_features[numer_features] = all_features[numer_features].fillna(0)

#用独热编码处理离散值
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(float)

# print(all_features.shape)

#数据转换
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32, device=device)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32, device=device)
#训练标签制作
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),dtype=torch.float32, device=device)



#定义损失函数
loss = nn.MSELoss()


#创建模型
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features,256),
        nn.ReLU(),
        nn.Linear(256,64),
        nn.ReLU(),
        nn.Linear(64,16),
        nn.ReLU(),
        nn.Linear(16,1)
    )
    #对每一层的权重进行 Xavier 初始化
    for layer in net:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
    return net.to(device)


#定义误差函数---对数均方根误差（相对误差）
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


#定义一个数据集加载器
def load_array(data_array, batch_size):
    features, labels = data_array
    dataset = TensorDataset(features,labels) #将特征张量和标签张量封装成一个 TensorDataset 对象
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True) #返回数据集加载器
    return data_loader


#定义loss函数
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        #记录训练损失
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


#k折交叉验证
def get_kf_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_kf_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        writer.add_scalar("train log_rmse",train_ls[-1],i+1)
        writer.add_scalar("valid log_rmse",valid_ls[-1],i+1)
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, ' f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


#添加tensorboard
writer = SummaryWriter("logs_train")

#设置超参数
k, num_epochs, lr, weight_decay, batch_size = 10, 100, 0.01, 15, 64
#查看k折训练误差来调整超参数
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {train_l}, ' f'平均验证log rmse: {float(valid_l):f}')

writer.close()
#进行正式模型预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    tain_ls, _ = train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    #查看log mrse
    print(f'训练log rmse：{float(tain_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0]) #Series 是一种类似于一维数组的数据结构，它由一组数据和与之相关联的索引（标签）组成。
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv',index=False) #保存结果


train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)
