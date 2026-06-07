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

    print("\n" + "=" * 60)
    print("过拟合分析")
    print("=" * 60)
    print(f"最终训练集 MSE Loss: {final_train_loss:.6f}")
    print(f"最终验证集 MSE Loss: {final_val_loss:.6f}")
    print(f"验证集/训练集 Loss 比值: {ratio:.4f}")
    print(f"Loss 差值 (Val - Train): {gap:.6f}")
    print(f"验证集最低 Loss: {best_val_loss:.6f} (Epoch {best_val_epoch})")

    if ratio > 1.5:
        print("\n⚠ 判断结果: 存在明显过拟合")
        print("  验证集 Loss 显著高于训练集 Loss，模型在训练数据上过度拟合。")
    elif ratio > 1.2:
        print("\n⚠ 判断结果: 存在轻微过拟合")
        print("  验证集 Loss 略高于训练集 Loss，模型有一定程度的过拟合。")
    else:
        print("\n✓ 判断结果: 未发现明显过拟合")
        print("  训练集和验证集 Loss 接近，模型泛化能力良好。")

    # 检查后期趋势
    last_50_train = train_losses[-50:]
    last_50_val = val_losses[-50:]
    val_increasing = last_50_val[-1] > min(last_50_val)
    train_decreasing = last_50_train[-1] < last_50_train[0]

    if val_increasing and train_decreasing:
        print("  注意: 训练后期验证集 Loss 有上升趋势，而训练集 Loss 仍在下降，")
        print("  这是过拟合的典型特征。建议使用早停 (Early Stopping) 策略。")

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
