import numpy as np
import pandas as pd
from torch.utils import data
import matplotlib.pyplot as plt
from torch import nn
import torch

#读取数据
train_data = pd.read_csv('kaggle_house_pred_train.csv')
test_data = pd.read_csv('kaggle_house_pred_test.csv')
features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) #第一个是ID，train的label要去除


#这里写一个处理数据的函数
def data_processing(data, measure):
    numeric_data = data.select_dtypes(include='number')
    categorical_data = data.select_dtypes(exclude='number')
    numeric_data = numeric_data.apply(measure)
    numeric_data = numeric_data.fillna(numeric_data.mean())
    categorical_data = pd.get_dummies(categorical_data, dummy_na=True)  #生成独热编码
    return pd.concat([numeric_data, categorical_data], axis=1)


#这里数据处理我们用均值方差来归一化
def measure(x):
    return (x - x.mean())/x.std()


#把所有的数据整理一下
features = data_processing(features, measure)
n_train = train_data.shape[0]

train_features = torch.tensor(features[:n_train].values.astype(np.float32))
test_features = torch.tensor(features[n_train:].values.astype(np.float32))
train_labels = torch.log(torch.tensor(train_data['SalePrice'].values.astype(np.float32)).reshape(-1, 1))


#网络
net = nn.Sequential(nn.Linear(train_features.shape[1], 1)) #模型的话，我并没有选择太复杂的模型


#损失函数
loss = nn.MSELoss()


#优化器
lr = 0.01
weight_decay = 0
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)



#写一个迭代器函数\
def load_data(data_arrays, batch_size):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


#写一个训练的函数
def train(net, optimizer, num_epochs, batch_size, train_features, train_labels, test_features, test_labels):
    train_ls = []
    test_ls = []
    data_iter = load_data((train_features, train_labels),batch_size)
    for epoch in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(l.item())
        if test_labels is not None:
            test_ls.append(loss(net(test_features), test_labels).item())
    return train_ls, test_ls


#K折交叉验证
def get_k_fold_data(k, i, X, y):
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


#开始训练
k = 5
batch_size = 64
num_epochs = 20
for i in range(5):
    X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_features, train_labels)
    train_ls, test_ls = train(net, optimizer, num_epochs, batch_size, X_train, y_train, X_valid, y_valid)
    print(f"第{i}折：train——{train_ls[-1]} , test——{test_ls[-1]}")


train_loss, _ = train(net, optimizer, num_epochs, batch_size
                      , train_features, train_labels, test_features, None)
plt.plot(torch.arange(num_epochs),train_loss, label='train')
plt.legend()
plt.show()
print(train_loss[-1])
preds = torch.exp(net(test_features)).detach().numpy()
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)