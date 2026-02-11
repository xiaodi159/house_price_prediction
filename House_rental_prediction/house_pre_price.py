#导库
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
from fontTools.subset import subset
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset      # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader         # 数据加载器.
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os

from numpy.ma.extras import median
from unicodedata import category
#更改文件路径
os.chdir('E:\\projects\\Art_learn')
# os.getcwd()

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
plt.rcParams["axes.unicode_minus"] = False



"""
数据预处理，生成 预训练/测试 数据
"""
def Data_Processing(data):
    #去除无关特征值
    data = data.drop(columns=['ID', '出租方式', '地铁站点', '小区名', '居住状态', '装修情况'])

    #删除小区信息为空的行
    data = data.dropna(subset=['位置','区'])

    # 添加新的特征值，表示是否有跌贴
    data["是否有地铁"] = data["距离"].notnull().astype(int)

    # 用最大距离填充“无地铁”的情况
    max_distance = data["距离"].max()
    data["距离"] = data["距离"].fillna(max_distance)

    #用平均值替换缺失值(主要替换小区售出数量）
    data['小区房屋出租数量'] = data['小区房屋出租数量'].fillna(data['小区房屋出租数量'].median())

    # num_cols = data.select_dtypes(include=np.number).columns
    # for col in num_cols:
    #     data[col] = data[col].fillna(data[col].median())

    #用字典将 房屋朝向 更改为数字类型代替
    unique_value =data['房屋朝向'].dropna().unique()
    map_dic = {value: idx for idx, value in enumerate(unique_value)}
    data['房屋朝向'] = data['房屋朝向'].map(map_dic)

    data["地铁线路"] = data["地铁线路"].fillna("无地铁")
    # One-Hot 编码
    data = pd.get_dummies(
        data,
        columns=["地铁线路"],
        prefix="地铁线"
    )

    x = data.drop(columns=['Label'])
    y = data['Label']

    feature_names = x.columns.tolist()
    # print(feature_names)

    #数据类型转换
    x = x.values.astype(np.float32)
    y = y.values.astype(np.float32)

    #数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #数据标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #数据集封装，转换为torch张量
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    return train_dataset, test_dataset, map_dic, feature_names, x_train.shape[1], scaler



class house_pre_price(nn.Module):
    def __init__(self, input_dim):
        #初始化父类
        super().__init__()

        self.Linear1 = nn.Linear(in_features=input_dim, out_features=256)
        # 批量归一化
        self.bn1 = nn.BatchNorm1d(256)
        #正则化
        self.dropout1 = nn.Dropout(0.4)
        self.Linear2 = nn.Linear(in_features=256, out_features=128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.Linear3 = nn.Linear(in_features=128, out_features=64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.4)
        self.output = nn.Linear(in_features=64, out_features=1)

    #定义前向传播函数
    def forward(self, x):
        x = torch.relu(self.bn1(self.Linear1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.Linear2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.Linear3(x)))
        x = self.dropout3(x)
        x = self.output(x)
        return x


def train(train_dataset, input_dim, epoch):
    model = house_pre_price(input_dim)
    model.train()
    # 数据集对象
    # 参1: 数据集对象, 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试集: 不打乱)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

    #优化损失函数
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    #L2 正则
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(epoch):
        total_loss, mean_loss = 0.0, 0.0
        start = time.time()
        for x, y in train_loader:

            y = y.view(-1, 1)

            y_pred = model(x)
            # print(y.shape, y_pred.shape)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch +1} Loss {mean_loss:.4f} Time {time.time() - start:.2f}s')

    #保存模型
    torch.save(model.state_dict(), './model/house_pre_price.pth')


def evaluate(test_dataset, input_dim, model_path):
    #加载模型
    model = house_pre_price(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True)

    # 定义评估指标
    # 平均绝对误差
    mae_loss = nn.L1Loss()
    # 均方误差
    mse_loss = nn.MSELoss()

    total_mae = 0.0
    total_mse = 0.0

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for x, y in test_loader:
            y = y.view(-1, 1)
            y_pred = model(x)

            total_mae += mae_loss(y_pred, y).item()
            total_mse += mse_loss(y_pred, y).item()

            # 保存预测值与真实值
            y_true_list.append(y.numpy())
            y_pred_list.append(y_pred.numpy())

    # 计算最终指标
    mean_mae = total_mae / len(test_loader)
    mean_mse = total_mse / len(test_loader)
    r_mse = np.sqrt(mean_mse)

    y_pred = np.vstack(y_pred_list)
    y_true = np.vstack(y_true_list)

    # R² = 1 - SSE / SST
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 打印预估结果
    print("========= 模型评估结果 =========")
    print(f"MAE  : {mean_mae:.4f}")
    print(f"MSE  : {mean_mse:.4f}")
    print(f"RMSE : {r_mse:.4f}")
    print(f"R²   : {r2:.4f}")


def Data_Processing_predict(data, map_dic, feature_names, scaler):
    idx = data['ID'].values

    # 去除无关特征（和训练一致）
    data = data.drop(columns=['ID', '出租方式', '地铁站点', '小区名', '居住状态', '装修情况'])

    # 是否有地铁
    data["是否有地铁"] = data["距离"].notnull().astype(int)

    # 距离填充
    max_distance = data["距离"].max()
    data["距离"] = data["距离"].fillna(max_distance)

    # 小区房屋出租数量
    data["小区房屋出租数量"] = data["小区房屋出租数量"].fillna(
        data["小区房屋出租数量"].median()
    )

    # 房屋朝向（用训练时的 map_dic）
    data["房屋朝向"] = data["房屋朝向"].map(map_dic)

    # 地铁线路 One-Hot（无地铁）
    data["地铁线路"] = data["地铁线路"].fillna("无地铁")
    data = pd.get_dummies(data, columns=["地铁线路"], prefix="地铁线")

    # 补齐缺失的特征列
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0

    # 按训练特征顺序重排
    data = data[feature_names]

    # 标准化（用训练 scaler）
    x = scaler.transform(data.values.astype(np.float32))

    #封装
    x= torch.tensor(x, dtype=torch.float32)

    return idx, x



def creat_data(idx, x_tensor, input_dim, model_path):
    model = house_pre_price(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(x_tensor)

    y_pred = y_pred.numpy().reshape(-1)

    # ===== 5. 生成结果并导出 CSV =====
    result_df = pd.DataFrame({
        "ID": idx,
        "Label": y_pred
    })

    result_df.to_csv('./data/rent_forecast/predict_result.csv',
                     index=False,
                     encoding='utf-8')




if __name__ == '__main__':
    train_data = pd.read_csv('./data/rent_forecast/train.csv', encoding='UTF-8')

    train_dataset, test_dataset, map_dic, feature_names, input_dim, scaler = Data_Processing(train_data)
    # x, y ,map_dic, feature_count = Data_Processing(train_data)
    # print(map_dic)
    # print(input_dim)

    #模型训练
    # train(train_dataset, input_dim, 50)

    #模型评估
    # evaluate(train_dataset, input_dim, './model/house_pre_price.pth')

    pre_data = pd.read_csv('./data/rent_forecast/test_noLabel.csv', encoding='UTF-8')

    idx, pre_data = Data_Processing_predict(pre_data, map_dic, feature_names, scaler)

    creat_data(idx, pre_data, input_dim, './model/house_pre_price.pth')


