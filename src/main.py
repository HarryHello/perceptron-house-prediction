"""
多层感知机 (MLP) 购房预测分类任务 - 主入口
使用 Kaggle House Prices 数据集，基于 PyTorch 实现 MLP 进行房价预测
"""

import os
import pandas as pd
import torch

from data import load_data, preprocess
from model import MLP
from train import train, evaluate, predict
from plot import plot_loss_curve, analyze_overfitting

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "house-prices-advanced-regression-techniques")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 1. 数据加载与预处理
    train_df, test_df = load_data(DATA_DIR)
    data = preprocess(train_df, test_df, batch_size=64, val_ratio=0.2)

    # 2. 创建模型
    model = MLP(data["input_dim"], hidden_dims=[128, 64, 32], dropout=0.2)
    print(f"\n模型结构:\n{model}")

    # 3. 训练
    train_losses, val_losses = train(model, data, num_epochs=300, lr=0.001)

    # 4. 绘制 loss 曲线
    plot_loss_curve(train_losses, val_losses, OUTPUT_DIR)

    # 5. 过拟合分析
    final_train_loss, final_val_loss = evaluate(model, data)
    analyze_overfitting(train_losses, val_losses, final_train_loss, final_val_loss, OUTPUT_DIR)

    # 6. 测试集预测
    test_preds = predict(model, data["X_test_tensor"])
    submission = pd.DataFrame({"Id": data["test_ids"], "SalePrice": test_preds.flatten()})
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"\n预测结果已保存至: {submission_path}")

    # 7. 保存模型
    model_path = os.path.join(OUTPUT_DIR, "mlp_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
