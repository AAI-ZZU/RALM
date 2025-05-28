import matplotlib.pyplot as plt  
import numpy as np  

# Font size parameters
TITLE_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 18
TEXT_FONT_SIZE = 14

# 数据，转换为百分数（乘以 100）
models = ['AM', '2D-Ptr', 'RANS']
hatches = ['/', '\\', '..']  # 使用与第二幅图一致的斜线填充
colors = ['#BAC8D3', '#F3D1A0', '#8da0cb']  # 统一颜色
min_costs = {
    'AM': [15.05, 35.27, 25.37],
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
width = 0.25  # 调整柱宽以匹配第二幅图

# 创建图形
fig, axs = plt.subplots(1, 2, figsize=(14, 7))  # 调整图形尺寸与第二幅图一致

# 绘制下限柱状图
for i, model in enumerate(models):
    bars = axs[0].bar(x + (i * width),
                      [min_costs[model][j] for j in range(len(scales))],
                      width, label=model, hatch=hatches[i], color=colors[i], edgecolor='black')  # 添加边框

    # 在柱状图上方添加数字
    for bar in bars:
        yval = bar.get_height()
        if yval >= 0:  # 正数
            axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}",
                         ha='center', va='bottom', fontsize=TEXT_FONT_SIZE)  # 使用文本字体大小参数
        else:  # 负数，紧贴柱下
            axs[0].text(bar.get_x() + bar.get_width() / 2, yval-0.1, f"{yval:.1f}",
                         ha='center', va='top', fontsize=TEXT_FONT_SIZE)

# 绘制上限柱状图
for i, model in enumerate(models):
    bars = axs[1].bar(x + (i * width),
                      [max_costs[model][j] for j in range(len(scales))],
                      width, label=model, hatch=hatches[i], color=colors[i], edgecolor='black')  # 添加边框

    # 在柱状图上方添加数字
    for bar in bars:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}",
                     ha='center', va='bottom', fontsize=TEXT_FONT_SIZE)  # 使用文本字体大小参数

# 设置下限柱状图
axs[0].set_xlabel('Number of Nodes', fontsize=LABEL_FONT_SIZE)  # 统一字体大小
axs[0].set_ylabel('Lower Bound of Optimality Gap (%)', fontsize=LABEL_FONT_SIZE)  # 统一字体大小
axs[0].set_xticks(x + width / 2)
axs[0].set_xticklabels(scales, fontsize=TICK_FONT_SIZE)  # 统一字体大小
axs[0].set_ylim(-5, 40)  # 设置Y轴范围：0到40
axs[0].legend(fontsize=TICK_FONT_SIZE)  # 统一字体大小
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# 设置上限柱状图
axs[1].set_xlabel('Number of Nodes', fontsize=LABEL_FONT_SIZE)  # 统一字体大小
axs[1].set_ylabel('Upper Bound of Optimality Gap (%)', fontsize=LABEL_FONT_SIZE)  # 统一字体大小
axs[1].set_xticks(x + width / 2 + 0.3)  # 向右移动x轴标签
axs[1].set_xticklabels(scales, fontsize=TICK_FONT_SIZE)  # 统一字体大小
axs[1].set_ylim(0, 400)  # 设置Y轴范围：0到400
axs[1].legend(fontsize=TICK_FONT_SIZE)  # 统一字体大小
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# 通过添加基线进行负数表现
axs[0].axhline(0, color='black', linewidth=0.8)
axs[1].axhline(0, color='black', linewidth=0.8)

# 调整布局
plt.tight_layout()

# 高清导出图片，设置更高的 DPI
plt.savefig('cost_comparison.pdf', dpi=4000, bbox_inches='tight')

# 显示图形
plt.show()