import numpy as np

def compute_loss(w, b, x, y):
    """
	计算二次损失函数
	
	参数：
	w (float): 权重
	b (float): 偏置
	x (float): 输入值
	y (float): 真实标签
	
	返回：
	float: 损失值
	"""
    y_pred = w * x + b
    loss = 0.5 * (y - y_pred) ** 2
    return loss

def compute_gradient(w, b, x, y):
    """
	计算损失函数对w和b的梯度
	参数：
	w (float): 权重
	b (float): 偏置
	x (float): 输入值
	y (float): 真实标签
	
	返回：
	tuple: 返回梯度（dw, db）
	"""
    y_pred = w * x + b
    dw = -(y - y_pred) * x
    db = -(y - y_pred)
    return dw, db

def gradient_descent(x, y, learning_rate=0.1, num_iterations=100):
    """
    基于梯度下降法优化w和b
    参数：
    x (array): 输入值数组
    y (array): 真实标签数组
    w_init (float): 初始权重
    b_init (float): 初始偏置
    learning_rate (float): 学习率
    num_iterations (int): 迭代次数

    返回：
    tuple: 最终的优化参数w和b
    """
    w, b = 0, 0
    loss_history = []

    for i in range(num_iterations):
        dw, db = 0, 0
        total_loss = 0
        # 计算所有样本梯度
        for j in range(len(x)):
            dw_i, db_i = compute_gradient(w, b, x[j], y[j])
            dw += dw_i
            db += db_i
            total_loss += compute_loss(w, b , x[j], y[j])
        dw /= len(x)
        db /= len(x)

        # update
        w -= learning_rate * dw
        b -= learning_rate * db

        # 计算平均损失并记录
        average_loss = total_loss / len(x)
        loss_history.append(average_loss)

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {average_loss:.4f}")
    return w, b, loss_history

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([3, 4, 7, 9, 11]) # y = 2*x + 1

w_final, b_final, loss_history = gradient_descent(x_data, y_data, learning_rate=0.1, num_iterations=100)

print(f"最终的权重: w = {w_final:.4f}, 偏置: b = {b_final:.4f}")
