#import "base/templates/report.typ": *
#import "base/functions/booktab.typ": booktab
#import "@preview/tablem:0.3.0": tablem, three-line-table

#show: report.with(
  title: "人工智能实验",
  subtitle: "基于多层感知机的购房预测",
  name: "马德全",
  stdid: "24325207",
  classid: "B201",
  major: "计算机科学与技术",
  school: "计算机学院",
  time: "2025~2026 学年第二学期",
  banner: "../images/sysu.png",
)

= 实验目的

通过本次实验，理解多层感知机（MLP）的基本原理与网络结构，掌握前向传播、损失函数和梯度下降优化方法，并使用 MLP 完成 Kaggle House Prices 购房预测任务。

= 实验原理

== 多层感知机

多层感知机（Multi-Layer Perceptron, MLP）是一种前馈神经网络，包含输入层、一个或多个隐藏层和输出层。每一层的神经元通过全连接方式与下一层相连，并通过激活函数引入非线性。

前向传播公式为：

$ h_1 = sigma_1(W_1 times x + b_1) $

$ y = sigma_2(W_2 times h_1 + b_2) $

其中 $W$ 为权重矩阵，$b$ 为偏置，$sigma$ 为激活函数。

== 激活函数

本实验使用 ReLU 作为隐藏层激活函数：

$ "ReLU"(x) = max(0, x) $

ReLU 计算简单、收敛速度快，能有效缓解梯度消失问题。

== 损失函数

本实验选择均方误差（MSE）作为损失函数：

$ L_"MSE" = 1/(2N) sum_(i=1)^N (y_i - hat(y)_i)^2 $

其中 $y_i$ 为真实房价的对数值，$hat(y)_i$ 为模型预测值。

== 梯度下降

采用 Adam 优化器进行参数更新：

$ theta = theta - eta nabla_theta L(theta) $

其中 $eta$ 为学习率，$nabla_theta L$ 为损失函数对参数的梯度。Adam 优化器结合了动量和自适应学习率，收敛更稳定。

= 实验内容与步骤

== 数据集

使用 Kaggle House Prices 数据集，包含 1460 条训练数据和 1459 条测试数据，共 80 项房屋特征（包括面积、质量、建造年份、地段等）。训练集进一步按 80\%/20\% 比例划分为训练集和验证集。

== 数据预处理

#set par(first-line-indent: 2em)

+ *数据过滤去噪*：原始数据中可能包含噪声数据。本实验进行了以下过滤：
  - 去除 `GrLivArea > 4000` 且 `SalePrice < 300000` 的异常点（大面积但低价的非典型住宅，共 2 条）
  - 去除 `LotArea >= 50000` 的极端地块（非常规用地，共 10 条）
  - 去除 `TotalBsmtSF > 3000` 且 `SalePrice < 300000` 的异常点（共 1 条）
  - 共移除 13 条异常数据（0.9\%），剩余 1447 条
+ *缺失值填充*：数值特征用中位数填充，类别特征用众数填充。
+ *类别编码*：使用 LabelEncoder 将类别特征转为数值。
+ *目标变量变换*：对 `SalePrice` 取对数（$log(1+y)$），使分布更接近正态。
+ *特征标准化*：使用 StandardScaler 对特征进行零均值单位方差标准化。

#set par(first-line-indent: 0em)

== 模型结构

#booktab(
  columns: (100pt, 100pt, 100pt),
  aligns: (center, center, center),
  caption: "MLP 网络结构",
  [层],
  [维度],
  [激活函数],
  [输入层],
  [79],
  [-],
  [隐藏层 1],
  [128],
  [ReLU + BatchNorm + Dropout(0.2)],
  [隐藏层 2],
  [64],
  [ReLU + BatchNorm + Dropout(0.2)],
  [隐藏层 3],
  [32],
  [ReLU + BatchNorm + Dropout(0.2)],
  [输出层],
  [1],
  [-],
)

== 训练配置

#booktab(
  columns: (120pt, 120pt),
  aligns: (center, center),
  caption: "训练超参数",
  [参数],
  [值],
  [损失函数],
  [MSE],
  [优化器],
  [Adam],
  [学习率],
  [0.001],
  [权重衰减],
  [$1 times 10^(-4)$],
  [学习率调度],
  [ReduceLROnPlateau (patience=20, factor=0.5)],
  [Batch Size],
  [64],
  [训练轮数],
  [300],
)

= 实验结果

== Loss 曲线

#figure(
  image("../src/output/loss_curve.png", width: 100%),
  caption: "训练与验证 Loss 曲线",
)

左图为线性坐标下的 Loss 曲线，右图为对数坐标。可以看到训练集和验证集的 Loss 均在前 100 个 Epoch 快速下降，之后趋于平缓。

== 过拟合分析

#figure(
  image("../src/output/overfitting_analysis.png", width: 80%),
  caption: "过拟合分析图",
)

#booktab(
  columns: (100pt, 100pt),
  aligns: (center, center),
  caption: "最终评估指标",
  [指标],
  [值],
  [训练集 MSE Loss],
  [0.2130],
  [验证集 MSE Loss],
  [0.2788],
  [验证集/训练集比值],
  [1.31],
  [验证集最低 Loss],
  [0.1321 (Epoch 269)],
)

分析结论如下：

+ *存在轻微过拟合*：验证集 Loss 略高于训练集 Loss，比值为 1.31，说明模型有一定程度的过拟合。
+ *后期趋势*：训练后期验证集 Loss 出现上升趋势，而训练集 Loss 仍在下降，这是过拟合的典型特征。
+ *Dropout 的作用*：模型使用了 Dropout(0.2) 正则化，在一定程度上缓解了过拟合，但未能完全消除。
+ *建议*：可采用早停（Early Stopping）策略，在验证集 Loss 不再下降时提前终止训练；或增大 Dropout 比率、减少隐藏层维度来进一步抑制过拟合。

= 实验总结

本实验基于 Kaggle House Prices 数据集，使用多层感知机（MLP）完成了购房预测任务。主要工作包括：

+ 对原始数据进行过滤去噪，移除了 13 条异常数据（0.9\%），包括大面积低价住宅、极端地块等噪声数据。
+ 使用 MSE 损失函数和 Adam 优化器训练了三层隐藏层的 MLP 模型，采用 BatchNorm 和 Dropout 正则化。
+ 通过训练集和验证集的 Loss 曲线对比分析，判断模型存在轻微过拟合，验证集/训练集 Loss 比值为 1.31。
+ 建议后续采用早停策略或增大正则化强度来进一步改善过拟合问题。
