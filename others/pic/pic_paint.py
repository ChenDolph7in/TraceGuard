import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前目录下所有的CSV文件
file_paths = [f for f in os.listdir() if f.endswith('.csv')]
labels = [os.path.splitext(f)[0] for f in file_paths]  # 提取文件名作为标签

# 创建一个图形
plt.figure(figsize=(10, 6))

# 循环读取每个CSV文件并绘制折线图
for file_path, label in zip(file_paths, labels):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 假设CSV文件中有列名为 'Step' 和 'Value'
    plt.plot(data['Step'], data['Value'], label=label)

# 设置横轴范围
plt.xlim(0, 19)

# 设置横轴刻度为每隔1个单位
plt.xticks(range(0, 20, 1))  # 从0到19，间隔为1

# 添加图形标题和标签
plt.title('3 Methods\' Training Loss Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()  # 显示图例
plt.grid(True)  # 添加网格
plt.savefig("./pic")
