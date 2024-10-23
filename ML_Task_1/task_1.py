import pandas as pd                                     # pandas用于数据处理和CSV文件操作。
import numpy as np                                      # numpy用于数学运算。
import torch                                             
import torch.nn as nn                                   # torch和torch.nn用于构建神经网络
import torch.optim as optim                             # torch.optim用于选择优化器。
from sklearn.model_selection import train_test_split    # train_test_split用于分割数据集。
from sklearn.preprocessing import StandardScaler        # StandardScaler用于特征归一化。
import matplotlib.pyplot as plt                         # matplotlib.pyplot用于数据可视化。

# 数据加载
data = pd.read_csv('C:/Users/21415/Desktop/machineLearning/ML_Task_1/train.csv') # 加载数据
print(data.head())    #head()方法默认返回DataFrame的前5行数据

# 数据探索
print(data.info())
print(data.describe())

# 数据清洗
data.fillna(data.mean(), inplace=True)  # 用均值填充缺失值

# 特征和目标变量提取
X = data.iloc[:, :-1].values  # 提取特征
y = data.iloc[:, -1].values   # 提取目标变量
# 特征归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义神经网络
class BostonHousingNN(nn.Module):
    def __init__(self):
        super(BostonHousingNN, self).__init__()
        self.fc1 = nn.Linear(X_scaled.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = BostonHousingNN()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
# 预测结果可视化
predictions = predictions.numpy()
y_test = y_test_tensor.numpy()

plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # 参考线
plt.show()
