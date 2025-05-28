import matplotlib.pyplot as plt
import numpy as np

# 数据准备
categories = ['With DCA', 'Without DCA']
with_dc_values = [65.25, 68.19]  # our and pomo for With DCA
without_dc_values = [64.38, 64.89]  # our and pomo for Without DCA

# 第二组数据
with_dc_values_2 = [57.67, 59.6694]  # our and pomo for With DCA
without_dc_values_2 = [56.60, 56.92]  # our and pomo for Without DCA

# 参数设置
bar_width = 0.15  # 柱宽设置
bar_edge_width = 1.0  # 柱边缘的宽度
font_size = 16  # 字体大小
data_label_offset = 0.5  # 数据标签的垂直偏移
legend_font_size = 16  # 图例字体大小
ylabel_font_size = 18  # Y轴文字大小
title_font_size = 14  # 标题字体大小
data_label_x_offset = -0.01  # 标签相对柱子的左右偏移
label_font_size = 22  # 使用的标签字体大小（可以根据需要调整）

# 横坐标位置
x = np.arange(len(categories))  # 每个分类的x位置

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# 第一张图: 3-500
axs[0].bar(x - bar_width/2, with_dc_values, width=bar_width, color='#BAC8D3', label='Original', edgecolor='black',
           linewidth=bar_edge_width, hatch='/')  # 加入斜线
axs[0].bar(x + bar_width/2, without_dc_values, width=bar_width, color='#F3D1A0', label='Refined', edgecolor='black',
           linewidth=bar_edge_width, hatch='\\')  # 加入斜线

# 添加数据标签
for i, val in enumerate(with_dc_values):
    axs[0].text(x[i] - bar_width/2 + data_label_x_offset, val + data_label_offset, f"{val:.2f}",
                ha='center', fontsize=font_size)  # 左侧柱的标签偏移
for i, val in enumerate(without_dc_values):
    axs[0].text(x[i] + bar_width/2 - data_label_x_offset, val + data_label_offset, f"{val:.2f}",
                ha='center', fontsize=font_size)  # 右侧柱的标签偏移

# 设置Y轴标签和范围
axs[0].set_ylabel('Cost (Min-Max Travel Time)', fontsize=ylabel_font_size)
axs[0].set_ylim(40, 80)  # 设置Y轴最大值为70
axs[0].set_xticks(x)
axs[0].set_xticklabels(categories, fontsize=legend_font_size)  # 控制X坐标轴字体大小
# 添加图例并移动位置
axs[0].legend(loc='upper left', fontsize=legend_font_size)

# 第二张图: 5-1000
axs[1].bar(x - bar_width/2, with_dc_values_2, width=bar_width, color='#BAC8D3', label='Original', edgecolor='black',
           linewidth=bar_edge_width, hatch='/')  # 加入斜线
axs[1].bar(x + bar_width/2, without_dc_values_2, width=bar_width, color='#F3D1A0', label='Refined', edgecolor='black',
           linewidth=bar_edge_width, hatch='\\')  # 加入斜线

# 添加数据标签
for i, val in enumerate(with_dc_values_2):
    axs[1].text(x[i] - bar_width/2 + data_label_x_offset, val + data_label_offset, f"{val:.2f}",
                ha='center', fontsize=font_size)  # 左侧柱的标签偏移
for i, val in enumerate(without_dc_values_2):
    axs[1].text(x[i] + bar_width/2 - data_label_x_offset, val + data_label_offset, f"{val:.2f}",
                ha='center', fontsize=font_size)  # 右侧柱的标签偏移

# 设置Y轴标签和范围
axs[1].set_ylabel('Cost (Min-Max Travel Time)', fontsize=ylabel_font_size)
axs[1].set_ylim(40, 70)  # 设置Y轴最大值为75
axs[1].set_xticks(x)
axs[1].set_xticklabels(categories, fontsize=legend_font_size)  # 控制X坐标轴字体大小
# 添加图例并移动位置
axs[1].legend(loc='upper left', fontsize=legend_font_size)

# 在第二个图的每个子图下方添加标签，向上移动位置
label_offset = -0.08  # 控制标签的Y轴偏移，这个参数可以调整上下位置
axs[0].text(0.5, label_offset, '(a) M3-N500', ha='center', va='top', fontsize=label_font_size, transform=axs[0].transAxes)  # 增加字体大小
axs[1].text(0.5, label_offset, '(b) M5-N1000', ha='center', va='top', fontsize=label_font_size, transform=axs[1].transAxes)  # 增加字体大小

# 设置整体布局并显示图形
plt.tight_layout()
# 高清导出图片，设置更高的 DPI
plt.savefig('op-cost-dca.pdf', dpi=4000, bbox_inches='tight')  # DPI 与第一幅图一致

plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据准备
# datasets = ['V3-U500', 'V5-U1000']
# with_dca = [65.10, 57.52]  # 成本
# without_dca = [68.09, 59.52]  # 成本
# with_dca_op = [63.89, 56.35]  # 最优成本
# without_dca_op = [64.72, 57.21]  # 最优成本
#
# # 计算gap
# with_dca_gap = [(b - a) / a * 100 for a, b in zip(with_dca_op, with_dca)]
# without_dca_gap = [(b - a) / a * 100 for a, b in zip(without_dca_op, without_dca)]
#
# # 设置参数
# bar_width = 0.15  # 柱宽
# x = np.arange(len(datasets))  # X坐标位置
#
# # 创建子图
# fig, ax = plt.subplots(figsize=(4, 4))  # 设置图形的宽度
#
# # 绘制柱状图
# bars1 = ax.bar(x - bar_width/2, with_dca_gap, width=bar_width, color='#BAC8D3',
#                edgecolor='black', hatch='/', label='With DCA')  # With DCA的gap
# bars2 = ax.bar(x + bar_width/2, without_dca_gap, width=bar_width, color='#F3D1A0',
#                edgecolor='black', hatch='\\', label='Without DCA')  # Without DCA的gap
#
# # 在每个柱子上添加数据标签
# vertical_offset = 0.05  # 控制数字的上下移动（向上偏移0.05个单位）
# horizontal_offset = 0.0  # 控制数字的左右移动（不移动）
#
# for bar in bars1:
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2 + horizontal_offset, yval + vertical_offset, f"{yval:.2f}",
#             ha='center', va='bottom', fontsize=10)  # 去掉了百分号
#
# for bar in bars2:
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2 + horizontal_offset, yval + vertical_offset, f"{yval:.2f}",
#             ha='center', va='bottom', fontsize=10)  # 去掉了百分号
#
# # 设置X轴和Y轴标签
# ax.set_ylabel('Gap (%)', fontsize=12)
# # ax.set_title('Cost Gap Comparison', fontsize=14)
# ax.set_xticks(x)
# ax.set_xticklabels(datasets, fontsize=12)
#
# # 添加Y轴范围设置
# ax.set_ylim(0, 8)  # 设置Y轴从0到8%
#
# # 添加图例
# ax.legend(fontsize=12)
#
# # 调整布局
# plt.tight_layout()
#
# # 显示图表
# plt.show()