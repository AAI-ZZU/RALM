import matplotlib.pyplot as plt
import numpy as np

# 数据，转换为百分数（乘以 100）
models = ['AM', '2D-Ptr', 'RANS']
hatches = ['/', '\\', '..']  # 将 "+" 替换为点 ".."，点密度更低
colors = ['#BAC8D3', '#F3D1A0', '#8da0cb']  # 更柔和且现代的颜色
min_costs = {
    'AM': [15.05, 35.27, 25.37],    # 小数转换为百分数
    '2D-Ptr': [2.41, 18.03, 22.95],
    'RANS': [-1.61, 4.33, 7.91]
}
max_costs = {
    'AM': [90.45, 236.90, 356.84],
    '2D-Ptr': [39.05, 102.02, 123.42],
    'RANS': [13.74, 15.82, 15.88]
}

# 规模标签
scales = ["[100-200)", "[200-500)", "[500-1000)"]

# 柱状图位置
x = np.arange(len(scales))  # 位置数组
width = 0.2  # 更细的柱宽

# 创建图形
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 调整图形尺寸与第一幅图一致

# 绘制下限柱状图
for i, model in enumerate(models):
    bars = axs[0].bar(x + (i * width),  # 改为紧靠在一起
                      [min_costs[model][j] for j in range(len(scales))],
                      width, label=model, hatch=hatches[i], color=colors[i])

    # 在柱状图上方添加数字
    for bar in bars:
        yval = bar.get_height()
        if yval >= 0:
            axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}",  # 去掉百分号
                         ha='center', va='bottom', fontsize=10)  # 字体大小与第一幅图一致
        else:
            axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}",  # 去掉百分号
                         ha='center', va='top', fontsize=10)  # 字体大小与第一幅图一致

# 绘制上限柱状图
for i, model in enumerate(models):
    bars = axs[1].bar(x + (i * width),
                      [max_costs[model][j] for j in range(len(scales))],
                      width, label=model, hatch=hatches[i], color=colors[i])

    # 在柱状图上方添加数字
    for bar in bars:
        yval = bar.get_height()
        if yval >= 0:
            axs[1].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}",  # 去掉百分号
                         ha='center', va='bottom', fontsize=10)  # 字体大小与第一幅图一致
        else:
            axs[1].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}",  # 去掉百分号
                         ha='center', va='top', fontsize=10)  # 字体大小与第一幅图一致

# 设置下限柱状图
axs[0].set_xlabel('Number of Nodes', fontsize=14)  # 字体大小与第一幅图一致
axs[0].set_ylabel('Lower Bound of Optimality Gap(%)', fontsize=14)  # 去掉百分号
axs[0].set_xticks(x + width / 2)
axs[0].set_xticklabels(scales, fontsize=14)  # 字体大小与第一幅图一致
axs[0].legend(fontsize=14)  # 字体大小与第一幅图一致
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# 设置上限柱状图
axs[1].set_xlabel('Number of Nodes', fontsize=14)  # 字体大小与第一幅图一致
axs[1].set_ylabel('Upper Bound of Optimality Gap(%)', fontsize=14)  # 去掉百分号
axs[1].set_xticks(x + width / 2)
axs[1].set_xticklabels(scales, fontsize=14)  # 字体大小与第一幅图一致
axs[1].legend(fontsize=14)  # 字体大小与第一幅图一致
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# 通过添加基线进行负数表现
axs[0].axhline(0, color='black', linewidth=0.8)  # 添加基线
axs[1].axhline(0, color='black', linewidth=0.8)  # 添加基线

# 调整布局
plt.tight_layout()

# 高清导出图片，设置更高的 DPI
plt.savefig('cost_comparison.pdf', dpi=4000, bbox_inches='tight')  # DPI 与第一幅图一致

# 显示图形
plt.show()