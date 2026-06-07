"""绘图与过拟合分析"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False


def plot_loss_curve(train_losses, val_losses, output_dir):
    """绘制训练和验证集的 loss 曲线"""
    num_epochs = len(train_losses)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, num_epochs + 1), train_losses, label="训练集 Loss", color="blue")
    axes[0].plot(range(1, num_epochs + 1), val_losses, label="验证集 Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("训练与验证 Loss 曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, num_epochs + 1), train_losses, label="训练集 Loss", color="blue")
    axes[1].plot(range(1, num_epochs + 1), val_losses, label="验证集 Loss", color="red")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss (log scale)")
    axes[1].set_title("训练与验证 Loss 曲线（对数坐标）")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nLoss 曲线已保存至: {path}")


def analyze_overfitting(train_losses, val_losses, final_train_loss, final_val_loss, output_dir):
    """过拟合分析并绘制分析图"""
    num_epochs = len(train_losses)
    ratio = final_val_loss / final_train_loss
    gap = final_val_loss - final_train_loss
    best_val_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = min(val_losses)

    # 绘制分析图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, num_epochs + 1), train_losses, label="训练集 Loss", color="blue", linewidth=1.5)
    ax.plot(range(1, num_epochs + 1), val_losses, label="验证集 Loss", color="red", linewidth=1.5)
    ax.axvline(x=best_val_epoch, color="green", linestyle="--", label=f"最佳 Epoch ({best_val_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("过拟合分析 - 训练与验证 Loss 对比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.annotate(f"Train: {final_train_loss:.4f}", xy=(num_epochs, final_train_loss),
                xytext=(-80, 20), textcoords="offset points", color="blue", fontsize=10)
    ax.annotate(f"Val: {final_val_loss:.4f}", xy=(num_epochs, final_val_loss),
                xytext=(-80, -25), textcoords="offset points", color="red", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "overfitting_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n过拟合分析图已保存至: {path}")
