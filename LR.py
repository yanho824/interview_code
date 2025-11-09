import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    m = len(y)
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
    return loss

def logistic_regression(X, y, learning_rate=0.01, num_iterations=100):
    m, n = X.shape # m:样本数 n:特征数
    w = np.zeros(n)
    b = 0
    loss_history = []

    for i in range(num_iterations):
        # 计算线性模型输出
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        
        # 计算损失
        loss = compute_loss(y, y_hat)
        loss_history.append(loss)

        # 计算梯度
        dw = np.dot(X.T, (y_hat - y)) / m
        db = np.mean(y_hat - y)

        w -= learning_rate * dw
        b -= learning_rate * db
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")
    return w, b, loss_history


X = np.array([[1, 2], [1, 3], [2, 3], [4, 5], [6, 7]])  # 输入特征
y = np.array([0, 0, 0, 1, 1])  # 真实标签

# 训练逻辑回归模型

w, b, loss_history = logistic_regression(X, y, learning_rate=0.1, num_iterations=1000)

print("训练完成的权重:", w)
print("训练完成的偏置:", b)