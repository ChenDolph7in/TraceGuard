import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前目录下所有的CSV文件
file_paths = [f for f in os.listdir() if f.endswith('.csv')]
labels = [os.path.splitext(f)[0] for f in file_paths]  # 提取文件名作为标签

# 用于存储不同参与比例的损失变化幅度（当前epoch的损失 - 前一个epoch的损失）
change_dict = {10: [], 30: [], 50: [], 70: [], 100: []}
# 时间消耗数据
time_consumptions = [33.0957, 80.3918, 101.4214, 157.5498, 221.1422]

# 循环读取每个CSV文件并提取数据
for file_path, label in zip(file_paths, labels):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 计算每个epoch的损失变化幅度（Loss(epoch) - Loss(epoch-1)）
    loss_changes = data['Value'].diff().dropna()  # diff() 计算当前与前一个epoch的差值

    # 计算每秒的损失变化量
    for loss_change in loss_changes:
        # 遍历不同的参与比例，按照文件名中包含的比例进行分组
        for i, percentage in enumerate([10, 30, 50, 70, 100]):
            # 对应的时间消耗
            time_consumption = time_consumptions[i]
            change_per_second = loss_change / time_consumption
            # 将变化量添加到对应比例的字典中
            if f'{percentage}percent' in label:
                change_dict[percentage].append(change_per_second)

# 准备数据用于箱型图
data_for_boxplot = [change_dict[10], change_dict[30], change_dict[50], change_dict[70], change_dict[100]]

# 创建箱型图
plt.figure(figsize=(10, 6))
plt.boxplot(data_for_boxplot, labels=[10, 30, 50, 70, 100])

# 添加标题和标签
plt.title('Loss Change Magnitude per Second by Participation Rate', fontsize=16)
plt.xlabel('Participation Rate (%)', fontsize=14)
plt.ylabel('Loss Change per Second (Loss Change / Time)', fontsize=14)

# 显示图像
plt.grid(True)
plt.savefig('./pic_boxplot_loss_change_per_second.png')
plt.show()
