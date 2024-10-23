### 一.机器学习基本概念

1. **监督学习**：​ 数据集包含输入和对应的输出标签。给算法数据集(input)，数据集中有正确答案，通过数据集训练模型(model),最后让算法通过自动学习进而可以给出更多的正确答案(output)。"Learns from being given 'right answers.'"  
<br>

2. **无监督学习**：数据集不包含输出标签。给定数据仅带有输入x而没有输出标签y，通过算法在数据中找到一定的结构或模式，由算法自己决定如何分组。  
<br>

3. **半监督学习**：使用少量标记数据和大量未标记数据进行训练。
<br>

4. **回归**：从无数的可能中预测一种可能的结果输出  
<br>

5. **分类**：预测一小部分可能的输出或类别


### 二.一些问题
#### 1).深度学习和机器学习的区别
- **机器学习**：
  - 包含广泛的算法和模型，依赖于特征工程。
  - 常用算法：线性回归、决策树、随机森林等。

- **深度学习**：
  - 基于神经网络的多层结构，自动提取特征。
  - 常用框架：TensorFlow、PyTorch、Keras等。
- **深度学习是机器学习的子集**
#### 2).偏导数、链式法则、梯度、矩阵等数学概念在机器学习中的作用
1. **偏导数**
   * **定义**：偏导数是指多变量函数相对于某一变量的导数，表示在其他变量保持不变时，该变量变化对函数值的影响。
   * **作用**：偏导数用于计算损失函数相对于模型参数的变化率
<br>

2. **链式法则**
   * **定义**：链式法则是求复合函数导数的一种方法，即将一个函数的导数分解为多个部分的导数的乘积。
   * **作用**：链式法用在反向传播算法中，计算神经网络中每层的梯度，从而更新权重。
<br>

3. **梯度**
   * **定义**：梯度是一个向量，包含了函数在各个方向上的偏导数，指向函数增长最快的方向。
   * **作用**：在优化过程中，梯度用于指示如何调整模型参数以减少损失函数。梯度下降算法利用梯度信息来迭代更新参数，逐步逼近最优解。
<br>

4. **矩阵**
   * **定义**：矩阵是一个二维数组，用于表示和处理数据。
   * **作用**：
        * **数据表示**：在机器学习中，数据通常以矩阵形式表示，特征矩阵（X）和标签向量（y）是基本构成。
        * **线性变换**：许多机器学习算法（如线性回归、支持向量机等）依赖于矩阵运算来进行线性变换和计算。
        * **并行计算**：矩阵运算可以利用GPU进行高效的并行计算，加速训练过程。

#### 3).常见的激活函数
- **Sigmoid**：输出范围在(0, 1)，常用于二分类问题。
    - **公式**：\( f(x) = \frac{1}{1 + e^{-x}} \)
    ```python
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    ```

- **Tanh**：输出范围在(-1, 1)，比 Sigmoid 更适合深层网络。
    - **公式**：\( f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
    ```python
    def tanh(x):
        return np.tanh(x)
    ```

- **ReLU（Rectified Linear Unit）**：输出为输入的正部分，计算效率高。
    - **公式**：\( f(x) = \max(0, x) \)
    ```python
    def relu(x):
        return np.maximum(0, x)
    ```

- **Softmax**：用于多分类问题，将输出转换为概率分布。
    - **公式**：\( f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} \)
    ```python
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # 为了数值稳定性
        return exp_x / exp_x.sum(axis=0, keepdims=True)
    ```

#### 4).神经网络基本结构
1. **输入层**
   - **功能**：接收外部输入数据。
   - **特点**：
     - 每个神经元对应输入特征的一个维度。
     - 不进行任何计算，只负责传递信息。

2. **隐藏层**
   - **功能**：进行数据处理和特征提取。
   - **特点**：
     - 可以有一个或多个隐藏层。
     - 每层由多个神经元组成，神经元之间通过权重连接。
     - 使用激活函数（如 ReLU、Sigmoid 等）引入非线性。

3. **输出层**
   - **功能**：生成最终的输出结果。
   - **特点**：
     - 输出层的神经元数量取决于具体任务（如二分类、多分类或回归）。
     - 常用的激活函数包括 Softmax（用于多分类）和 Sigmoid（用于二分类）。

4. **权重和偏置**
   - **权重**：连接神经元的参数，控制信息传递的强度。
   - **偏置**：在每个神经元中添加的额外参数，帮助模型更好地拟合数据。

5. **激活函数**
   - **功能**：决定神经元是否激活。
   - **常见激活函数**：
     - Sigmoid
     - Tanh
     - ReLU
     - Softmax

6. **损失函数**
   - **功能**：衡量模型预测与真实值之间的差距。
   - **常见损失函数**：
     - 均方误差（MSE）用于回归问题。
     - 交叉熵损失用于分类问题。

7. **优化算法**
   - **功能**：更新权重和偏置以最小化损失函数。
   - **常见优化算法**：
     - 随机梯度下降（SGD）
     - Adam
     - RMSprop
  
#### 5).机器学习中的数据处理
1. **数据收集**
   - **功能**：获取原始数据。
   - **来源**：可以通过数据库、API、爬虫、传感器等方式收集数据。

2. **数据清洗**
   - **功能**：处理缺失值、重复值和异常值。
   - **方法**：
     - **缺失值处理**：
       - 删除含缺失值的样本或特征。
       - 使用均值、中位数、众数填充。
       - 使用插值法。
     - **重复值处理**：删除重复记录。
     - **异常值处理**：通过统计方法（如 Z-score、IQR）识别并处理异常值。

3. **数据转换**
   - **功能**：将数据转换为适合模型处理的格式。
   - **方法**：
     - **特征缩放**：
       - 标准化（Z-score normalization）：将特征转换为均值为0，标准差为1的分布。
       - 归一化（Min-Max scaling）：将特征缩放到[0, 1]区间。
     - **特征编码**：
       - 类别变量编码（如独热编码、标签编码）。
       - 文本数据处理（如词袋模型、TF-IDF）。
  
4. **特征选择与提取**
   - **功能**：选择对模型有用的特征，减少维度。
   - **方法**：
     - **特征选择**：
       - 过滤方法（如卡方检验）。
       - 包裹方法（如递归特征消除）。
       - 嵌入方法（如基于模型的特征重要性）。
     - **特征提取**：
       - 主成分分析（PCA）。
       - 线性判别分析（LDA）。

5. **数据划分**
   - **功能**：将数据集分为训练集、验证集和测试集。
   - **方法**：
     - 随机划分。
     - 交叉验证（如K折交叉验证）。

6. **数据增强**
   - **功能**：增加训练数据的多样性，防止过拟合。
   - **方法**：
     - 对图像数据进行旋转、翻转、缩放等变换。
     - 对文本数据进行同义词替换、随机插入等。

#### 6).欠拟合和过拟合
1.  **欠拟合**：泛化能力差，训练样本集准确率低，测试样本集准确率低。
  **过拟合**：泛化能力差，训练样本集准确率高，测试样本集准确率低。
  **合适的拟合程度**：泛化能力强，训练样本集准确率高，测试样本集准确率高
<br>

2. **欠拟合原因：**
    - 训练样本数量少
    - 模型复杂度过低
    - 参数还未收敛就停止循环

3. **欠拟合的解决办法：**
    - 增加样本数量
    - 增加模型参数，提高模型复杂度
    - 增加循环次数
    - 查看是否是学习率过高导致模型无法收敛
  4. **过拟合原因：**
     - 数据噪声太大
     - 特征太多
     - 模型太复杂
4. **过拟合的解决办法：**
    - 清洗数据
    - 减少模型参数，降低模型复杂度
    - 增加惩罚因子（正则化），保留所有的特征，但是减少参数的大小（magnitude）。
#### 7). 神经网络中的正则化

1. **什么是正则化**：一种防止过拟合的技术。正则化是通过给损失函数添加一些限制，来规范模型的学习过程，防止模型在训练集上过度拟合，从而提高其在测试集上的泛化能力。

2. **为什么要进行正则化**：
   - **防止过拟合**：当模型在训练集上学习过多细节时，可能会失去对新数据的泛化能力。正则化可以限制模型的复杂性，帮助其更好地适应未见过的数据。

3. **怎么进行正则化**：
   - **L1 正则化**（Lasso）：在损失函数中添加权重绝对值的和，促使一些权重变为零，从而实现特征选择。
     \[
     \text{Loss} = \text{Loss}_{\text{original}} + \lambda \sum |w_i|
     \]
   - **L2 正则化**（Ridge）：在损失函数中添加权重平方和，促使权重尽可能小，避免模型过于复杂。
     \[
     \text{Loss} = \text{Loss}_{\text{original}} + \lambda \sum w_i^2
     \]
   - **Dropout**：在训练过程中随机丢弃一定比例的神经元，以减少模型对特定神经元的依赖，从而增强模型的泛化能力。
   - **数据增强**：通过对训练数据进行变换（如旋转、平移、缩放等），增加数据的多样性，从而减少过拟合的风险。
   - **早停法**（Early Stopping）：在验证集上监控模型的性能，当性能不再提升时停止训练，以防止过拟合。

#### 8).线性回归模型
线性回归是用来建立**特征**与**目标**之间的线性关系，模型表示为y = b + x1 * w1 + x2 * w2 + ...，目标是找到最佳的参数b, w1, w2...，常使用均方误差（MSE）作为损失函数。
下面是一个简单的线性回归模型
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 4.0, 6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    loss = (y_pred - y) ** 2
    return loss

w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(-2.0, 2.1, 0.1)

# mse_matrix用于存储不同 w,b 组合下的均方误差损失
mse_matrix = np.zeros((len(w_list), len(b_list)))

for i, w in enumerate(w_list):
    for j, b in enumerate(b_list):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, w, b)
        mse_matrix[i, j]= l_sum/len(x_data)
W, B = np.meshgrid(w_list, b_list)
fig = plt.figure('Linear Model Cost Value')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, mse_matrix.T, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('loss')
plt.show()
```
#### 9).逻辑回归模型
逻辑回归是用来建立特征与目标之间的非线性关系，模型表示为 \( P(Y=1) = \frac{1}{1 + e^{-(b + x_1w_1 + x_2w_2 + \ldots)}} \)，目标是找到最佳的参数 \( b, w_1, w_2, \ldots \)，常使用对数损失（Log Loss）或交叉熵损失作为损失函数。
```python
import torch
# import torch.nn.functional as F
 
# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
 
#design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
 
    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()
 
# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
 
# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```
#### 10).梯度下降和随机梯度下降
梯度下降是一个迭代优化算法，通过不断调整参数来最小化损失函数，直到找到最优解。
梯度下降公式：θ:=θ−α∇J(θ)，即更新参数为当前参数减去学习率乘以当前梯度。
随机梯度下降是梯度下降的一种特殊情况，每次迭代只选取**一个样本**来计算梯度，然后更新参数。
梯度下降
```python
import matplotlib.pyplot as plt
 
# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# initial guess of weight 
w = 1.0
 
# define the model linear model y = w*x
def forward(x):
    return x*w
 
#define the cost function MSE 
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred - y)**2
    return cost / len(xs)
 
# define the gradient function  gd
def gradient(xs,ys):
    grad = 0
    for x, y in zip(xs,ys):
        grad += 2*x*(x*w - y)
    return grad / len(xs)
 
epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w-= 0.01 * grad_val  # 0.01 learning rate
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)
 
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show() 
```

随机梯度下降
```python
import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = 1.0
 
def forward(x):
    return x*w
 
# calculate loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
# define the gradient function  sgd
def gradient(x, y):
    return 2*x*(x*w - y)
 
epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        grad = gradient(x,y)
        w = w - 0.01*grad    # update weight by every grad of sample of training set
        print("\tgrad:", x, y,grad)
        l = loss(x,y)
    print("progress:",epoch,"w=",w,"loss=",l)
    epoch_list.append(epoch)
    loss_list.append(l)
 
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show() 
```
#### 11).mini-batch（小批量梯度下降）
把整个训练集划分成多个小批量的样本集，每次使用一小部分样本计算梯度。
#### 12).反向传播算法
在线性模型中，我们很容易用梯度下降来更新参数，但是在多层神经网络中，就需要前馈和反馈的过程。当前馈过程计算出loss后，再进行反馈，通过链式法则逐层计算损失函数对每个参数的偏导数。反向传播的目标是计算损失函数相对于每个参数的梯度，首先计算损失函数对输出层的激活值的梯度，再利用链式法则，逐层向后计算每个隐藏层的梯度，最后再根据计算出的梯度更新模型参数。

```python
# 如果是复杂的网络，没办法都自己写gradient的计算。
import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x, w):
    return x * w

def loss(x, y, w):
    y_pred = forward(x, w)
    loss = (y - y_pred) ** 2
    return loss

print('predict (before training)', 4, forward(4, w.item()))

epoch_list = []
loss_list = []

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y, w)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    print('process:', epoch, l.item())
    epoch_list.append(epoch)
    loss_list.append(l.item())

print('predict (after training)', 4, forward(4, w))

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

```
#### 13).多分类问题
将二分类变成多分类，要使用到softmax()函数