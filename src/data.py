"""数据加载与预处理"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(data_dir):
    """加载 Kaggle House Prices 数据集"""
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    print(f"训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")
    return train_df, test_df


def remove_outliers(train_df):
    """去除噪声数据/异常值"""
    original_len = len(train_df)

    # 1. 去除 GrLivArea > 4000 且 SalePrice 较低的异常点
    #    这是 Kaggle House Prices 数据集中已知的离群点
    #    (大面积但低价的商业/农业用地，不属于典型住宅)
    train_df = train_df.drop(
        train_df[(train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)].index
    )

    # 2. 去除 LotArea 极端异常值（面积超过 50000 平方英尺的非常规地块）
    train_df = train_df[train_df["LotArea"] < 50000]

    # 3. 去除 TotalBsmtSF 极端异常值（地下室面积超过 3000 且价格异常低）
    train_df = train_df[
        ~((train_df["TotalBsmtSF"] > 3000) & (train_df["SalePrice"] < 300000))
    ]

    removed = original_len - len(train_df)
    print(f"去噪: 移除 {removed} 条异常数据 ({removed/original_len*100:.1f}%), 剩余 {len(train_df)} 条")
    return train_df.reset_index(drop=True)


def preprocess(train_df, test_df, batch_size=64, val_ratio=0.2, random_state=42):
    """数据预处理：去噪、缺失值填充、编码、标准化，返回 DataLoader 和相关数据"""
    test_ids = test_df["Id"]

    # 数据过滤去噪
    train_df = remove_outliers(train_df)

    # 合并处理（方便统一编码）
    all_df = pd.concat([train_df.drop("SalePrice", axis=1), test_df], axis=0)
    all_df = all_df.drop("Id", axis=1)

    # 数值特征用中位数填充
    num_cols = all_df.select_dtypes(include=[np.number]).columns
    all_df[num_cols] = all_df[num_cols].fillna(all_df[num_cols].median())

    # 类别特征用众数填充
    cat_cols = all_df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        all_df[col] = all_df[col].fillna(all_df[col].mode()[0] if not all_df[col].mode().empty else "None")

    # LabelEncoder 编码
    for col in cat_cols:
        le = LabelEncoder()
        all_df[col] = le.fit_transform(all_df[col].astype(str))

    # 分离训练集和测试集
    X_all = all_df.values
    n_train = train_df.shape[0]
    X = X_all[:n_train]
    y = np.log1p(train_df["SalePrice"].values.reshape(-1, 1))

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=random_state
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 转 Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集
    X_test = scaler.transform(X_all[n_train:])
    X_test_tensor = torch.FloatTensor(X_test)

    return {
        "train_loader": train_loader,
        "train_dataset": train_dataset,
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_val_tensor": X_val_tensor,
        "y_val_tensor": y_val_tensor,
        "X_test_tensor": X_test_tensor,
        "test_ids": test_ids,
        "input_dim": X_train.shape[1],
        "scaler": scaler,
    }
