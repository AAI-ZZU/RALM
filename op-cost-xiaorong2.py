import matplotlib.pyplot as plt
import numpy as np

# 数据准备
labels = ['V3-U500', 'V5-U1000']  # 更新 X 轴标签
with_dc_gap = [-11.63, -5.73]  # With DC 的 gap
without_dc_gap = [-7.83, -3.33]  # Without DC 的 gap

x = np.arange(len(labels))  # 标签的位置
width = 0.15  # 调整柱子宽度

# 进一步减少柱子的 x 轴位置，使它们更靠近中间
bar1 = x - 0.075  # 向左移动柱子位置
bar2 = x + 0.075  # 向右移动柱子位置

# 绘制柱状图，并调整图表的大小
fig, ax = plt.subplots(figsize=(4, 4))  # 设置图形大小

a1 = ax.bar(bar1, with_dc_gap, width, label='With DCA', color='#BAC8D3', hatch='/')
a2 = ax.bar(bar2, without_dc_gap, width, label='Without DCA', color='#F3D1A0', hatch='\\')

# 添加一些文本标签
ax.set_ylabel('Gap (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)  # 设置 X 轴标签为 V3-U500 和 V5-U1000
ax.legend()

# 添加数据标签到柱子下方，并设置字体大小
for bar in a1:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -10),  # 向下偏移10个点
                textcoords="offset points",
                ha='center', va='top',
                fontsize=10)  # 减小字体大小

for bar in a2:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2 + 0.08, height),  # 更改为向右偏移0.08
                xytext=(0, -10),  # 向下偏移10个点
                textcoords="offset points",
                ha='center', va='top',
                fontsize=10)  # 减小字体大小

# 设置 x 轴范围，缩短空间
plt.xlim(-0.5, 1.5)  # 限制 x 轴范围以减小中间的空间

# 显示图表
plt.ylim([-15, 5])  # 设置 y 轴范围
plt.axhline(0, color='black', linewidth=0.8)  # 添加 y 轴基线
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格
plt.tight_layout(pad=1.0)  # 自动调整布局，减少空白
plt.savefig('op-cost-xiaorong.pdf', format='pdf', dpi=4000)  # 设置格式为PDF，DPI设置为4000以保持质量

plt.show()