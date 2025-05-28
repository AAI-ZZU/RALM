# coding=utf-8
# @Author : LJR
# @Time :  2025/1/28 21:52
# @Description :
import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['AM', 'ET', '2D-Ptr', 'DRL-li', 'LRAM']
before_optimization = [164.04, 183.692, 64.732, 60.48, 54.06]
after_optimization = [60.6741, 75.9310, 58.9611, 57.8517, 57.1207]

# 柱状图的位置
x = np.arange(len(models)) * 1.5  # 增加间距，乘以1.5

# 绘制柱状图
plt.figure(figsize=(6, 5))  # 调整图形尺寸，使其更宽
bar_width = 0.4  # 微调柱状宽度

# 使用高级的颜色和斜线填充
before_bar = plt.bar(x - bar_width / 2, before_optimization, bar_width, label='Original',
                     color='#BAC8D3', hatch='/', edgecolor='black')  # 温和的灰蓝色和斜线填充
after_bar = plt.bar(x + bar_width / 2, after_optimization, bar_width, label='Refined',
                    color='#F3D1A0', hatch='\\', edgecolor='black')  # 温和的淡金色和斜线填充

# 添加标签和标题
plt.xlabel('Model', fontsize=14)  # 设置x轴标签字体大小
plt.ylabel('Cost (Min-Max Travel Time)', fontsize=14)  # 设置y轴标签字体大小
plt.yticks(fontsize=14)  # 设置y刻度字体大小
plt.xticks(x, models, fontsize=14)  # 设置x刻度字体大小

# 增加X轴的间距
plt.xlim(-0.5, len(models) * 1.5 - 0.5)  # 调整X轴范围，以增加左右间距

# 添加图例
plt.legend(fontsize=14)  # 设置图例字体大小

# 在柱状图上添加数值
for i in range(len(models)):
    # 原始值
    plt.text(x[i] - bar_width / 2, before_optimization[i] + 2,
             f'{before_optimization[i]:.1f}', ha='center', fontsize=10)  # 保留一位小数，调整字体大小为10

    # 优化后的值 - 将数字稍微向右移动
    plt.text(x[i] + bar_width / 2 + 0.05, after_optimization[i] + 2,  # 加了0.1的偏移
             f'{after_optimization[i]:.1f}', ha='center', fontsize=10)  # 保留一位小数，调整字体大小为10

# 设置布局
plt.tight_layout()

# 高清导出图片，设置更高的 DPI
plt.savefig('op-cost_plot-our-xiaorong.pdf', dpi=1080, bbox_inches='tight')  # 设置分辨率，确保无多余空白区域

# 显示图形
plt.show()