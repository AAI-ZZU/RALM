import matplotlib.pyplot as plt
import numpy as np

# Data for ranks at different scales
ranks = {
    1000: {
        'ET': 274.86,
        'AM': 466.20,
        '2D-Ptr': 713.22,
        'DRL-li': 731.26,
    },
    500: {
        'ET': 225.54,
        'AM': 253.02,
        '2D-Ptr': 381.46,
        'DRL-li': 390.95,
    },
    300: {
        'ET': 165.72,
        'AM': 159.86,
        '2D-Ptr': 241.98,
        'DRL-li': 243.42,
    },
    200: {
        'ET': 121.76,
        'AM': 111.14,
        '2D-Ptr': 166.30,
        'DRL-li': 169.34,
    },
}

# Node sizes
node_sizes = sorted(ranks.keys())
# Models
models = list(ranks[200].keys())

# Calculate proportions and prepare data for plotting
proportions = {model: [] for model in models}

for size in node_sizes:
    for model in models:
        proportions[model].append(ranks[size][model] / size)

# X-axis values for models
x_labels = ['200', '300', '500', '1000']
x_values = np.arange(len(x_labels))  # [0, 1, 2, 3]

# Plotting - Making the figure very narrow
plt.figure(figsize=(4, 5))  # Adjusted to match the first plot's size

# Define different marker styles
markers = ['o', 'p', 'D', '^']  # 圆点、五边形、菱形、三角形

# Define colors and swap AM and 2D-Ptr
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 更高级的调色板

for i, model in enumerate(models):
    plt.plot(x_values, proportions[model], marker=markers[i], label=model, linewidth=1, linestyle='--', color=colors[i])

# Setting the x-ticks and labels
plt.xticks(x_values, x_labels, fontsize=14)  # 设置x刻度字体大小

# Adding titles and labels
plt.xlabel('Number of Nodes (N)', fontsize=14)  # 设置x轴标签字体大小
plt.ylabel('Rank / N+1', fontsize=14)  # 设置y轴标签字体大小
plt.yticks(fontsize=14)  # 设置y刻度字体大小
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Adding a legend
plt.legend(fontsize=14)  # 设置图例字体大小

# Set y-axis limits
plt.tight_layout()
plt.ylim(0.2, max(max(proportions[model]) for model in models) + 0.1)  # Set lower limit to 0.2

# 高清导出图片，设置更高的 DPI
# 高清导出PDF文件
plt.savefig('plot-rank.pdf', format='pdf', dpi=4000)  # 设置格式为PDF，DPI设置为4000以保持质量
plt.show()