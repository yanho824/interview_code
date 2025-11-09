import numpy as np

def binary_cross_entropy(y, y_pred):
    epsilon = 1e-15
    # 防止y_pred = 0 / 1
    # log(0) == -inf, log(1) == 0 梯度消失
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return np.mean(loss)

y = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])

loss = binary_cross_entropy(y, y_pred)
print(f"Binary cross entropy Loss: {loss:.4f}")

def categorical_cross_entropy(y, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

    loss = -np.sum(y * np.log(y_pred), axis=1)
    return np.mean(loss)

# 示例：真实标签和预测概率（one-hot编码）
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1], [0.8, 0.1, 0.1]])

# 计算交叉熵损失
loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy Loss: {loss:.4f}")