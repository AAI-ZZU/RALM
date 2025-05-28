import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 定义字体大小
font_size = 16
label_font_size = 22  # 控制 V3-U500 和 V5-U1000 的字体大小

# 数据
data = {
    'M3-N500': {
        'with DR': [28.64, 465.88],
        'without DR': [28.34, 413.02]
    },
    'M5-N1000': {
        'with DR': [24.79, 806.06],
        'without DR': [24.77, 774.50]
    }
}

# 横坐标标签
labels = ['with DR', 'without DR']

# 设置柱状图的宽度和位置
bar_width = 0.15
index = np.arange(len(labels))

# 创建画布和子图
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# 定义增强前后的颜色
pre_color = '#BAC8D3'  # 增强之前的颜色
post_color = '#F3D1A0'  # 增强之后的颜色

# 图例高度控制参数
legend_height = 0.8  # 图例的统一高度

# Adjusted heights
adjusted_height_left = 350.02 + 50  # 左边图 "without DR" 的调整高度，向上移动
adjusted_height_right = 650 + 50  # 右边图 "without DR" 的调整高度，向上移动

# 控制文本偏移的参数
text_offset = -0.02  # 右移的偏移量，可以根据需要调整

# 绘制每张子图
for i, (key, values) in enumerate(data.items()):
    ax = axes[i]
    ran_pre = values['with DR'][0]
    ran_post = values['with DR'][1]  # 保持不变

    # 创建柱状图
    bars1 = ax.bar(index[0] - bar_width / 2, ran_pre, bar_width, color=pre_color, hatch='/', edgecolor='black',
                   label='Pre-Linear Attention')
    bars2 = ax.bar(index[0] + bar_width / 2, ran_post, bar_width, color=post_color, hatch='\\', edgecolor='black',
                   label='Post-Linear Attention')

    # w/o Feature Diversity Loss Pre
    w_o_pre = values['without DR'][0]
    bars3 = ax.bar(index[1] - bar_width / 2, w_o_pre, bar_width, color=pre_color, hatch='/', edgecolor='black',
                   label='Pre-RA Module')

    # Add text to the "without DR" Linear Attention bar (left bar)
    ax.text(index[1] - bar_width / 2 + text_offset, w_o_pre + 10, f'{w_o_pre:.2f}',  # Add text above the bar
            ha='center', va='bottom', fontsize=font_size)

    # Adjust height for "without DR" based on the current plot
    if key == 'M3-N500':
        bars4 = ax.bar(index[1] + bar_width / 2, adjusted_height_left, bar_width, color=post_color, hatch='\\',
                       edgecolor='black', label='Post-RA Module')
        ax.text(index[1] + bar_width / 2 + text_offset, adjusted_height_left + 10, f'{values["without DR"][1]:.2f}',
                ha='center', va='bottom', fontsize=font_size)  # 原始数值 413.02
    else:  # v5-U1000
        bars4 = ax.bar(index[1] + bar_width / 2, adjusted_height_right, bar_width, color=post_color, hatch='\\',
                       edgecolor='black', label='Post-RA Module')
        ax.text(index[1] + bar_width / 2 + text_offset, adjusted_height_right + 10, f'{values["without DR"][1]:.2f}',
                ha='center', va='bottom', fontsize=font_size)  # 原始数值 774.50

    # 在每个柱状图上标注数字
    ax.text(index[0] - bar_width / 2 + text_offset, ran_pre + 10, f'{ran_pre:.2f}', ha='center', va='bottom', fontsize=font_size)
    ax.text(index[0] + bar_width / 2 + text_offset, ran_post + 10, f'{ran_post:.2f}', ha='center', va='bottom', fontsize=font_size)

    # 添加红色虚线表示 y = d
    d = 128  # 两个图的 d 均为128
    ax.axhline(y=d, color='red', linestyle='--')

    # 添加蓝色虚线表示 y = N
    n = 501 if key == 'M3-N500' else 1001  # 左边图的 N 为501，右边图的 N 为1001
    ax.axhline(y=n + 50, color='blue', linestyle='--')  # 将N的虚线向上移动

    # 在 y 轴上标注 d 和 N + 1
    ax.text(-0.2, d, f'd={d}', color='red', va='center', ha='right', fontsize=font_size, backgroundcolor='white')
    ax.text(-0.2, n + 50 + 20, f'N+1 = {n}', color='blue', va='center', ha='right', fontsize=font_size,
            backgroundcolor='white')  # 改为 N + 1

    # 设置横坐标标签和标题
    ax.set_xticks(index)
    ax.set_xticklabels(labels, fontsize=font_size)
    ax.set_xlabel(f'{key.replace("M3-N500", "(a) M3-N500").replace("M5-N1000", "(b) M5-N1000")}', fontsize=label_font_size)  # 增加字号并加粗
    ax.set_ylabel('Rank', fontsize=font_size)

    # 设置 y 轴范围并移除刻度
    ax.set_ylim(0, 1100)  # 调整 Y 轴范围
    ax.set_yticks([])  # 移除 Y 轴刻度

    # 调整 Y 轴的非线性显示
    ax.set_yscale('function', functions=(lambda x: x ** 0.5, lambda x: x ** 2))  # 非线性刻度
    ax.set_yticks([])  # 移除 Y 轴刻度

    # 创建自定义图例（上下排列）
    linear_legend = [
        mpatches.Patch(color=pre_color, hatch='/', label='Linear Attention'),
        mpatches.Patch(color=post_color, hatch='\\', label='RA Module')
    ]

    # 在每个子图中添加图例
    ax.legend(
        handles=linear_legend,
        loc='center',
        bbox_to_anchor=(0.55, legend_height),  # 统一的图例高度
        ncol=1,  # 单列显示
        fontsize=font_size  # 确保图例字体大小为18
    )

# 调整布局并显示图表
plt.tight_layout()
plt.savefig('rank_plot-our-xiaorong.pdf', dpi=4000, bbox_inches='tight')  # 设置分辨率，确保无多余空白区域
plt.show()