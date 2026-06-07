"""训练与评估"""

import torch
import torch.nn as nn


def train(model, data, num_epochs=300, lr=0.001, weight_decay=1e-4):
    """训练模型，返回训练和验证的 loss 记录"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    train_loader = data["train_loader"]
    train_dataset = data["train_dataset"]
    X_val_tensor = data["X_val_tensor"]
    y_val_tensor = data["y_val_tensor"]

    train_losses = []
    val_losses = []

    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
            val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]  "
                  f"Train Loss: {epoch_train_loss:.6f}  "
                  f"Val Loss: {val_loss:.6f}")

    print("训练完成!")
    return train_losses, val_losses


def evaluate(model, data):
    """在 eval 模式下评估训练集和验证集 Loss"""
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(data["X_train_tensor"]), data["y_train_tensor"]).item()
        val_loss = criterion(model(data["X_val_tensor"]), data["y_val_tensor"]).item()
    return train_loss, val_loss


def predict(model, X_test_tensor):
    """在测试集上生成预测"""
    model.eval()
    with torch.no_grad():
        preds_log = model(X_test_tensor).numpy()
    import numpy as np
    return np.expm1(preds_log)
