# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    # 加载数据
    X_train, y_train = read_data()
    # 添加偏置项
    m, n = X_train.shape
    X_train = np.column_stack((np.ones((m, 1)), X_train))
    n += 1  # 因为加了一列全为1的向量
    # 岭回归
    alpha = 0.1  # 正则化系数
    L = np.eye(n)  # L为n*n的单位矩阵
    L[0, 0] = 0  # 因为第一项代表偏置项，不需要正则化
    theta = np.linalg.inv(X_train.T.dot(X_train) + alpha * L).dot(X_train.T).dot(y_train)
    # 预测
    data = np.insert(data, 0, 1)  # 插入偏置项
    pred = np.dot(theta, data)
    return pred

def lasso(data):
    # 加载数据
    X_train, y_train = read_data()
    # 添加偏置项
    m, n = X_train.shape
    X_train = np.column_stack((np.ones((m, 1)), X_train))
    n += 1  # 因为加了一列全为1的向量
    # Lasso回归
    alpha = 0.015  # 正则化系数
    n_iter = 150000  # 迭代次数
    eta = 1e-12  # 学习率
    theta = np.zeros(n)  # 初始化模型参数
    for _ in range(n_iter):  # 迭代求解
        gradient = (1/m) * X_train.T.dot(X_train.dot(theta) - y_train) + alpha * np.sign(theta)
        theta -= eta * gradient
    # 预测
    data = np.insert(data, 0, 1)  # 插入偏置项
    pred = np.dot(theta, data)
    return pred

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
