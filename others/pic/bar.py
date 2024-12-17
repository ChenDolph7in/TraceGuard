import matplotlib.pyplot as plt

# 不同参与比例 (%)
participation_rates = [10, 30, 50, 70, 100]

# 对应的时间消耗（秒）
time_consumptions = [33.0957, 80.3918, 101.4214, 157.5498, 221.1422]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制柱状图
plt.bar(participation_rates, time_consumptions, color='skyblue', width=8, label='Time Consumption (s)')

# 在柱子上显示时间消耗的具体数值
for i, time in enumerate(time_consumptions):
    plt.text(participation_rates[i], time + 5, f'{time:.2f}s', ha='center', fontsize=12)

# 绘制折线图来展示时间消耗的变化趋势
plt.plot(participation_rates, time_consumptions, color='orange', marker='o', linestyle='-', linewidth=2, markersize=8, label='Time Trend')

# 添加标题和标签
plt.title('Time Consumption vs Participation Rate', fontsize=16)
plt.xlabel('Participation Rate (%)', fontsize=14)
plt.ylabel('Time Consumption (seconds)', fontsize=14)

# 设置横轴刻度并在 10, 30, 50, 70, 100 位置标记
plt.xticks(participation_rates)

# 显示图例
plt.legend()

# 保存图像
plt.savefig('participation_time_consumption_trend_with_xticks.png')
