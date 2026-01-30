"""
    房价预测


"""
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset      # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader         # 数据加载器.
from sklearn.preprocessing import StandardScaler, LabelEncoder



def creat_data():
    data = pd.read_csv('./data/HousePrice.csv', encoding='utf-8')
    # print(data.head())
    #特征类别
    # print(data.info())

    #损失值处理
    #统计每列缺失值个数，全0->无缺失值
    miss_count = data.isnull().sum()
    # print(miss_count)
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        mean_value = data[col].mean()
        data[col] = data[col].fillna(mean_value)

    #类型转换，将object类型转换为数字类型代替
    cat_cols = data.select_dtypes(include=['object']).columns
    # print(cat_cols)

    # 用于保存所有列的映射关系
    category_maps = {}

    for col in cat_cols:
        unique_calues = data[col].dropna().unique()
        # enumerate:
        # 为每个类别分配一个整数编号
        # 如 {'RL':0, 'RM':1, 'FV':2}
        mapping_dict = {value: idx for idx, value in enumerate(unique_calues)}
        #保存映射字字典
        category_maps[col] = mapping_dict
        #对数值进行映射
        data[col] = data[col].map(mapping_dict)
        data[col] = data[col].fillna(-1)

    #特征提取
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    #添加保存输入特征类别
    feature_names = x.columns.tolist()

    # values: DataFrame → ndarray
    x = x.values.astype(np.float32)
    y = y.values.astype(np.float32)

    #数据划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

    #数据标准化处理(转换器）
    # (优化） 分别对特征和目标进行标准化
    transfer_x = StandardScaler()
    x_train = transfer_x.fit_transform(x_train)
    x_test = transfer_x.transform(x_test)

    transfer_y = StandardScaler()
    y_train = transfer_y.fit_transform(y_train.reshape(-1, 1))
    y_test = transfer_y.transform(y_test.reshape(-1, 1))
    #把 y 从“二维数组”变成“一维浮点数组”
    y_train = y_train.astype(np.float32).ravel()
    y_test = y_test.astype(np.float32).ravel()


    # print(x_train.shape, x_test.shape)

    #数据封装
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    #返回值：训练集，测试集，（映射，字典），输入特征值数字
    #添加输出，特征类别， 转换器(x,和y）
    return train_dataset, test_dataset, category_maps, x_train.shape[1], feature_names, transfer_x, transfer_y



class ANN_house_price(nn.Module):
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
    model = ANN_house_price(input_dim)
    model.train()
    # 数据集对象
    # 参1: 数据集对象(1600条), 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试集: 不打乱)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    #优化损失函数
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    #L2 正则
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

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
    torch.save(model.state_dict(), './model/ANN_house_price.pth')

def evaluate(test_dataset, input_dim):
    #加载模型参数
    model = ANN_house_price(input_dim)
    model.load_state_dict(torch.load('./model/ANN_house_price.pth'))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    #定义评估指标
    #平均绝对误差
    mae_loss = nn.L1Loss()
    #均方误差
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

            #保存预测值与真实值
            y_true_list.append(y.numpy())
            y_pred_list.append(y_pred.numpy())

    #计算最终指标
    mean_mae = total_mae / len(test_loader)
    mean_mse = total_mse / len(test_loader)
    r_mse = np.sqrt(mean_mse)

    y_pred = np.vstack(y_pred_list)
    y_true = np.vstack(y_true_list)

    # R² = 1 - SSE / SST
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    #打印预估结果
    print("========= 模型评估结果 =========")
    print(f"MAE  : {mean_mae:.4f}")
    print(f"MSE  : {mean_mse:.4f}")
    print(f"RMSE : {r_mse:.4f}")
    print(f"R²   : {r2:.4f}")


def predict_house_price(input_dim, feature_names, scaler_x, scaler_y, category_maps):
    model = ANN_house_price(input_dim)
    model.load_state_dict(torch.load('./model/ANN_house_price.pth'))
    model.eval()

    print("\n请输入房屋特征（按提示输入）：")
    user_input = []

    for feature in feature_names:
        if feature in category_maps:
            mapping = category_maps[feature]
            print(f"\n特征：{feature}")
            for k, v in mapping.items():
                print(f"  {k} → {v}")
            val = input("请输入类别名称：")
            while val not in mapping:
                val = input("请输入类别名称：")
            user_input.append(mapping[val])
        else:
            val = float(input(f"{feature}（数值型）："))
            user_input.append(val)

    # 转 numpy
    user_input = np.array(user_input).reshape(1, -1).astype(np.float32)

    # X 标准化
    user_input = scaler_x.transform(user_input)

    with torch.no_grad():
        pred_std = model(torch.tensor(user_input))

    # ★★★ 反标准化 ★★★
    pred_price = scaler_y.inverse_transform(pred_std.numpy())

    print("\n==============================")
    print(f"🏠 预测房价为：{pred_price[0][0]:,.0f}")
    print("==============================")





if __name__ == '__main__':
    train_dataset, test_dataset, category_maps, input_dim, feature_names, scaler_x, scaler_y = creat_data()
    # print(category_maps)
    # print(input_dim)
    # print(feature_names)
    # model = ANN_house_price(input_dim)
    # print(model)

    #模型训练
    # train(train_dataset, input_dim, 50)

    #模型评估
    # evaluate(test_dataset, input_dim)

    #模型预测
    # 用户交互预测
    predict_house_price(input_dim, feature_names, scaler_x, scaler_y, category_maps)




