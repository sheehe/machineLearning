import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 检查是否有可用的CUDA设备（即GPU），如果有，则使用GPU进行计算，否则使用CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
batch_size = 64  # 每个batch的大小
learning_rate = 0.001  # 学习率
num_epochs = 10  # 训练的轮数

# 数据预处理：将图像转换为Tensor并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),                                  # 将PIL图像或NumPy数组转换为Tensor。
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 数据归一化，均值和标准差均为0.5
])

# 数据集加载
# 创建训练数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data',       # 数据存储的根目录
    train=True,          # 指定为训练集
    download=True,       # 如果数据集不存在，则自动下载
    transform=transform  # 应用之前定义的转换
)

# 创建训练数据加载器
trainloader = DataLoader(
    trainset,            # 传入训练数据集
    batch_size=batch_size,  # 每个批次的样本数
    shuffle=True         # 在每个 epoch 开始时打乱数据
)

# 创建测试数据集
testset = torchvision.datasets.CIFAR10(
    root='./data',       # 数据存储的根目录
    train=False,         # 指定为测试集
    download=True,       # 如果数据集不存在，则自动下载
    transform=transform  # 应用之前定义的转换
)

# 创建测试数据加载器
testloader = DataLoader(
    testset,             # 传入测试数据集
    batch_size=batch_size,  # 每个批次的样本数
    shuffle=False        # 不打乱测试数据
)

# 定义神经网络结构
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入3通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32通道到64通道
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义最大池化层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 64通道，8x8大小，连接到512个神经元
        self.fc2 = nn.Linear(512, 10)  # 输出10个类别

    def forward(self, x):
        # 定义前向传播过程
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积 + 激活 + 池化
        x = x.view(-1, 64 * 8 * 8)  # 将特征图展平
        x = F.relu(self.fc1(x))  # 全连接层和激活函数
        x = self.fc2(x)  # 输出层
        return x

# 模型、损失函数和优化器
model = Network().to(device)  # 实例化模型并移动到设备上
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 用于保存训练过程中的损失和准确率
train_losses = []  # 训练损失列表
train_accuracies = []  # 训练准确率列表

def train():
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):  # 遍历每个epoch
        running_loss = 0.0  # 初始化当前epoch的损失
        correct = 0  # 初始化正确预测数量
        total = 0  # 初始化总样本数量
        for i, data in enumerate(trainloader, 0):  # 遍历训练数据加载器
            inputs, labels = data  # 获取输入和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 移动到设备

            optimizer.zero_grad()  # 清零梯度

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()  # 累加损失

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 更新总样本数量
            correct += (predicted == labels).sum().item()  # 更新正确预测数量

        # 计算当前epoch的准确率
        accuracy = 100 * correct / total
        train_losses.append(running_loss / len(trainloader))  # 保存损失
        train_accuracies.append(accuracy)  # 保存准确率
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test():
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化正确预测数量
    total = 0  # 初始化总样本数量
    with torch.no_grad():  # 不需要计算梯度
        for data in testloader:  # 遍历测试数据加载器
            images, labels = data  # 获取输入和标签
            images, labels = images.to(device), labels.to(device)  # 移动到设备
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 更新总样本数量
            correct += (predicted == labels).sum().item()  # 更新正确预测数量

    # 计算并打印测试集准确率
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

# 可视化训练过程
def plot_training_history():
    plt.figure(figsize=(12, 5))  # 设置图形大小

    # 绘制损失曲线
    plt.subplot(1, 2, 1)  # 创建1行2列的子图，当前为第1个子图
    plt.plot(train_losses, label='Training Loss')  # 绘制训练损失曲线
    plt.title('Training Loss')  # 设置标题
    plt.xlabel('Epochs')  # 设置x轴标签
    plt.ylabel('Loss')  # 设置y轴标签
    plt.legend()  # 显示图例

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)  # 当前为第2个子图
    plt.plot(train_accuracies, label='Training Accuracy')  # 绘制训练准确率曲线
    plt.title('Training Accuracy')  # 设置标题
    plt.xlabel('Epochs')  # 设置x轴标签
    plt.ylabel('Accuracy (%)')  # 设置y轴标签
    plt.legend()  # 显示图例

    plt.show()  # 显示图形

if __name__ == "__main__":
    train()  # 调用训练函数
    test()   # 调用测试函数
    plot_training_history()  # 绘制训练过程的损失和准确率曲线
