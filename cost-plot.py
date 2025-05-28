import numpy as np
import matplotlib.pyplot as plt

# 原始数据
ranks = {
    1000: {'ET': 1006, 'AM': 388.30, '2D-Ptr': 144.88, 'DRL-li': 102.45},
    500: {'ET': 261.295166015625, 'AM': 142.70, '2D-Ptr': 57.35, 'DRL-li': 50.80},
    300: {'ET': 145.56, 'AM': 70.62547, '2D-Ptr': 33.61, 'DRL-li': 31.948719},
    200: {'ET': 121.76, 'AM': 40.50, '2D-Ptr': 22.83, 'DRL-li': 22.31},
}

# 节点大小
node_sizes = sorted(ranks.keys())
# 模型
models = list(ranks[200].keys())

# 计算比例并准备绘图数据
proportions = {model: [] for model in models}

for size in node_sizes:
    for model in models:
        proportions[model].append(ranks[size][model])

# X轴标签
x_labels = ['200', '300', '500', '1000']
x_values = np.arange(len(x_labels))  # [0, 1, 2, 3]

# 实现平滑曲线
def smooth_curve(arr, window_len=2):
    """对数据进行简单平滑处理"""
    window = np.ones(window_len) / window_len
    return np.convolve(arr, window, mode='same')

# 绘图 - 设置图形尺寸，让图更瘦
plt.figure(figsize=(4, 5))  # 宽度缩小为4，高度保持不变

# 定义不同的标记样式和颜色
markers = ['o', 'p', 'D', '^']  # 圆点、五边形、菱形、三角形
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 高级颜色组合

# 绘制每个模型的曲线
for i, model in enumerate(models):
    if model == 'ET':
        smoothed_data = smooth_curve(proportions[model], window_len=2)  # 仅对ET进行平滑处理
        plt.plot(x_values, smoothed_data, marker=markers[i], label=model, linewidth=1, linestyle='--', color=colors[i])

    else:
        plt.plot(x_values, proportions[model], marker=markers[i], label=model, linewidth=1, linestyle='--', color=colors[i])

# 设置x轴刻度和标签
plt.xticks(x_values, x_labels, fontsize=14)  # 设置x刻度字体大小

# 添加标题和标签
plt.xlabel('Number of Nodes (N)', fontsize=14)  # 设置x轴标签字体大小
plt.ylabel('Cost (Min-Max Travel Time)', fontsize=14)  # 设置y轴标签字体大小
plt.yticks(fontsize=14)  # 设置y刻度字体大小
plt.grid(visible=True, linestyle='--', alpha=0.7)

# 使用对数刻度
plt.yscale('log')

# 设置Y轴刻度标签
y_ticks = [10, 20, 50, 100, 200, 500, 1000]  # 自定义Y轴刻度
plt.yticks(y_ticks, labels=[str(tick) for tick in y_ticks])  # 设置刻度和标签

# 添加图例
plt.legend(fontsize=14)  # 设置图例字体大小

# 设置布局
plt.tight_layout()

# 高清导出图片，设置更高的 DPI
plt.savefig('cost-plot.pdf', format='pdf', dpi=4000)  # 设置格式为PDF，DPI设置为4000以保持质量

# 显示图形
plt.show()