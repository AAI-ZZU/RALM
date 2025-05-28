import numpy as np
import matplotlib.pyplot as plt

# ------ 参数调整接口 ------
# 1. 整体图尺寸
FIG_WIDTH = 24  # 整体图的宽度
FIG_HEIGHT = 7  # 整体图的高度

# 2. 第一幅图（Cost and Rank Comparison）参数
RANK_Y_LIM = (0.3, 1)  # 柱状图的 Y 轴范围
COST_Y_PADDING = 50  # 折线图 Y 轴比最高值高出的部分
COST_LABEL_OFFSET_X = 0.12  # 折线图标注的水平偏移量
COST_LABEL_OFFSET_Y = 10  # 折线图标注的垂直偏移量
RANK_LABEL_OFFSET_Y = 0.02  # 柱状图标注的垂直偏移量
X_AXIS_PADDING = 0.9  # X 轴范围比数据点多出的部分
COST_LABEL_FONTSIZE = 12  # 折线图标注数字的字体大小（可以调整这个参数来改变大小）

# 3. 第二幅图（Optimization Comparison）参数
OPTIMIZED_BAR_WIDTH = 0.4  # 柱状图的宽度
OPTIMIZED_LABEL_OFFSET_Y = 2  # 柱状图标注的垂直偏移量

# ------ 第一幅图的数据和配置 ------
ranks = {
    1000: {'ET': 274.86, 'AM': 466.20, '2D-Ptr': 713.22, 'DRL-li': 731.26, 'LRAM': 814.26},
    500: {'ET': 225.54, 'AM': 253.02, '2D-Ptr': 381.46, 'DRL-li': 390.95, 'LRAM': 419.88},
    300: {'ET': 165.72, 'AM': 159.86, '2D-Ptr': 241.98, 'DRL-li': 243.42, 'LRAM': 256.84},
    200: {'ET': 121.76, 'AM': 111.14, '2D-Ptr': 166.30, 'DRL-li': 169.34, 'LRAM': 174.96}
}

costs = {
    1000: {'ET': 1006, 'AM': 388.30, '2D-Ptr': 144.88, 'DRL-li': 102.45, 'LRAM': 92.02},
    500: {'ET': 261.30, 'AM': 142.70, '2D-Ptr': 57.35, 'DRL-li': 50.80, 'LRAM': 46.65},
    300: {'ET': 145.56, 'AM': 70.63, '2D-Ptr': 33.61, 'DRL-li': 31.95, 'LRAM': 29.31},
    200: {'ET': 121.76, 'AM': 40.50, '2D-Ptr': 22.83, 'DRL-li': 22.31, 'LRAM': 20.50}
}

models = list(ranks[200].keys())
x_values = np.arange(len(models))  # 定义 x_values，表示每个模型的位置
colors = ['#E2F1D5', '#D0E4F0', '#F8C0C0', '#FAE8B0', '#6B5B95']
patterns = ['/', '\\', 'x', '+', '|']

# ------ 第二幅图的数据和配置 ------
models_optimized = ['AM', 'ET', '2D-Ptr', 'DRL-li', 'LRAM']
before_optimization = [164.04, 183.692, 64.732, 60.48, 54.06]
after_optimization = [60.6741, 75.9310, 58.9611, 57.8517, 57.1207]

# ------ 创建整体图的布局 ------
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))  # 整体图的尺寸
gs = fig.add_gridspec(2, 6)  # 2行6列布局

# ------ 第一幅图：Cost and Rank Comparison（左侧，占据前4列） ------
for idx, scale in enumerate(sorted(ranks.keys())):
    rank_values = [ranks[scale][model] / scale for model in models]
    cost_values = [costs[scale][model] for model in models]

    # ------ Cost 折线图 ------
    ax_cost = fig.add_subplot(gs[0, idx])  # 第一行的前4列
    ax_cost.plot(x_values, cost_values, linestyle='--', marker='o', markersize=8, linewidth=2, color='black', zorder=100,
                 markeredgecolor='black', markerfacecolor='black')  # Set color to black

    # 调整标注位置：在节点右上侧，紧挨着节点
    for i, value in enumerate(cost_values):
        if value == 1006:  # 仅将 1006 的位置向下移动
            ax_cost.text(
                x_values[i] + COST_LABEL_OFFSET_X, value - 10, f'{value:.1f}',  # 向下移动10个单位
                fontsize=COST_LABEL_FONTSIZE, ha='left', color='black', zorder=200
            )
        else:
            ax_cost.text(
                x_values[i] + COST_LABEL_OFFSET_X, value + COST_LABEL_OFFSET_Y, f'{value:.1f}',
                fontsize=COST_LABEL_FONTSIZE, ha='left', color='black', zorder=200
            )

    # 动态设置 Y 轴范围：比最高值稍高一点
    max_cost = max(cost_values)
    ax_cost.set_ylim(0, max_cost + COST_Y_PADDING)  # 设置 Y 轴范围

    # 设置更细的 Y 轴刻度
    y_ticks = np.linspace(0, max_cost + COST_Y_PADDING, num=15)  # 15 个刻度
    ax_cost.set_yticks(y_ticks)

    # 拉宽 X 轴范围，确保 LRAM 的值不会溢出
    ax_cost.set_xlim(-X_AXIS_PADDING, len(models) - 1 + X_AXIS_PADDING)

    ax_cost.set_title(f'Scale: {scale}', fontsize=14, fontweight='bold')
    ax_cost.set_xticks(x_values)
    ax_cost.set_xticklabels(models, fontsize=14, rotation=30)  # Set rotation for x-tick labels
    ax_cost.set_ylabel('Cost (Min-Max Travel Time)', fontsize=14, color='black')  # Change label color to black
    ax_cost.grid(True, linestyle=':')

    # ------ Rank 柱状图 ------
    ax_rank = fig.add_subplot(gs[1, idx])  # 第二行的前4列
    bars = ax_rank.bar(x_values, rank_values, color=colors, alpha=0.5, width=0.5, zorder=1, edgecolor='black',
                       linewidth=0.5)
    for j, bar in enumerate(bars):
        bar.set_hatch(patterns[j % len(patterns)])
    for i, value in enumerate(rank_values):
        ax_rank.text(
            x_values[i], value + RANK_LABEL_OFFSET_Y, f'{value:.2f}',
            fontsize=14, ha='center', color='black'
        )

    ax_rank.set_xticks(x_values)
    ax_rank.set_xticklabels(models, fontsize=14, rotation=30, ha='right')  # 旋转标签，避免重叠
    ax_rank.set_ylabel('Rank/N', fontsize=14)
    ax_rank.set_ylim(RANK_Y_LIM)  # 设置 Y 轴范围
    ax_rank.grid(True, linestyle=':')

# ------ 第二幅图：Optimization Comparison（右侧，占据最后2列） ------
ax_optimized = fig.add_subplot(gs[:, 4:])  # 整体占据最后两列
x = np.arange(len(models_optimized)) * 1.5
before_bar = ax_optimized.bar(
    x - OPTIMIZED_BAR_WIDTH / 2, before_optimization, OPTIMIZED_BAR_WIDTH,
    label='Original', color='#BAC8D3', hatch='/', edgecolor='black'
)
after_bar = ax_optimized.bar(
    x + OPTIMIZED_BAR_WIDTH / 2, after_optimization, OPTIMIZED_BAR_WIDTH,
    label='Refined', color='#F3D1A0', hatch='\\', edgecolor='black'
)

ax_optimized.set_ylabel('Cost (Average travel time for each vehicle)', fontsize=14)
ax_optimized.set_xticks(x)
ax_optimized.set_xticklabels(models_optimized, fontsize=14)
ax_optimized.legend(fontsize=14)
ax_optimized.grid(True, linestyle=':')

# 标注柱状图数值
for i in range(len(models_optimized)):
    ax_optimized.text(
        x[i] - OPTIMIZED_BAR_WIDTH / 2, before_optimization[i] + OPTIMIZED_LABEL_OFFSET_Y,
        f'{before_optimization[i]:.1f}', ha='center', fontsize=14
    )
    ax_optimized.text(
        x[i] + OPTIMIZED_BAR_WIDTH / 2, after_optimization[i] + OPTIMIZED_LABEL_OFFSET_Y,
        f'{after_optimization[i]:.1f}', ha='center', fontsize=14
    )

# ------ 在图的底部添加Figure (a) 和 Figure (b) ------
fig.text(0.35, 0.00, 'Figure (a)', ha='center', fontsize=16, fontweight='bold')  # 调整y值
fig.text(0.85, 0.00, 'Figure (b)', ha='center', fontsize=16, fontweight='bold')  # 调整y值

# ------ 调整整体布局 ------
plt.tight_layout()

# ------ 保存为PDF ------
plt.savefig('combined_plot.pdf', format='pdf', bbox_inches='tight', dpi=4000)

# ------ 显示图形 ------
plt.show()



'''
(parco) root@I1ee26c426500201d0f:/PADR/p8/parco-main# python test.py --problem hcvrp --decode_type greedy --batch_size 1
/usr/local/miniconda3/envs/parco/lib/python3.10/site-packages/torchrl/data/replay_buffers/samplers.py:37: UserWarning: Failed to import torchrl C++ binaries. Some modules (eg, prioritized replay buffers) may not work with your installation. If you installed TorchRL from PyPI, please report the bug on TorchRL github. If you installed TorchRL locally and/or in development mode, check that you have all the required compiling packages.
  warnings.warn(EXTENSION_WARNING)
Loading checkpoint from  ./checkpoints/hcvrp/parco.ckpt
/usr/local/miniconda3/envs/parco/lib/python3.10/site-packages/lightning/pytorch/utilities/parsing.py:209: Attribute 'env' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['env'])`.
/usr/local/miniconda3/envs/parco/lib/python3.10/site-packages/lightning/pytorch/utilities/parsing.py:209: Attribute 'policy' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['policy'])`.
val_file not set. Generating dataset instead
test_file not set. Generating dataset instead
Loading ./data/hcvrp/n10000_m3_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [1:31:52<00:00, 28.71s/it]
Average cost-1: 6003.2914
Average cost *8: 5304.4136
Per step inference time: 28.7091s
Total inference time: 5512.1378s
Average eval steps: 8544.41
Loading ./data/hcvrp/n10000_m5_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [1:08:29<00:00, 21.40s/it]
Average cost-1: 4589.6587
Average cost *8: 3414.6663
Per step inference time: 21.3983s
Total inference time: 4108.4795s
Average eval steps: 6339.65
Loading ./data/hcvrp/n10000_m7_seed24610_8.npz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [56:57<00:00, 17.80s/it]
Average cost-1: 3721.2863
Average cost *8: 2720.0347
Per step inference time: 17.7920s
Total inference time: 3416.0701s
Average eval steps: 5270.65
Loading ./data/hcvrp/n1000_m3_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [45:48<00:00,  2.68s/it]
Average cost-1: 483.7589
Average cost *8: 407.8638
Per step inference time: 2.6827s
Total inference time: 2747.0564s
Average eval steps: 871.20
Loading ./data/hcvrp/n1000_m5_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [39:36<00:00,  2.32s/it]
Average cost-1: 331.3729
Average cost *8: 253.0359
Per step inference time: 2.3193s
Total inference time: 2375.0096s
Average eval steps: 751.72
Loading ./data/hcvrp/n1000_m7_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [36:24<00:00,  2.13s/it]
Average cost-1: 203.6431
Average cost *8: 151.2719
Per step inference time: 2.1321s
Total inference time: 2183.2496s
Average eval steps: 690.08
Loading ./data/hcvrp/n200_m3_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [09:02<00:00,  1.89it/s]
Average cost-1: 23.0369
Average cost *8: 21.4466
Per step inference time: 0.5285s
Total inference time: 541.2150s
Average eval steps: 171.83
Loading ./data/hcvrp/n200_m5_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [08:48<00:00,  1.94it/s]
Average cost-1: 12.4115
Average cost *8: 12.2135
Per step inference time: 0.5150s
Total inference time: 527.3302s
Average eval steps: 166.62
Loading ./data/hcvrp/n200_m7_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [08:29<00:00,  2.01it/s]
Average cost-1: 8.7663
Average cost *8: 8.6400
Per step inference time: 0.4959s
Total inference time: 507.7802s
Average eval steps: 161.93
Loading ./data/hcvrp/n3000_m3_seed24610_8.npz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [24:21<00:00,  7.61s/it]
Average cost-1: 1674.6923
Average cost *8: 1416.6184
Per step inference time: 7.6088s
Total inference time: 1460.8855s
Average eval steps: 2474.90
Loading ./data/hcvrp/n3000_m5_seed24610_8.npz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [18:57<00:00,  5.92s/it]
Average cost-1: 1192.6503
Average cost *8: 902.7575
Per step inference time: 5.9201s
Total inference time: 1136.6663s
Average eval steps: 1919.94
Loading ./data/hcvrp/n3000_m7_seed24610_8.npz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [17:04<00:00,  5.33s/it]
Average cost-1: 1087.7052
Average cost *8: 759.5256
Per step inference time: 5.3323s
Total inference time: 1023.7984s
Average eval steps: 1732.99
Loading ./data/hcvrp/n5000_m3_seed24610_8.npz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [41:50<00:00, 13.07s/it]
Average cost-1: 2785.7632
Average cost *8: 2446.9815
Per step inference time: 13.0725s
Total inference time: 2509.9270s
Average eval steps: 4142.19
Loading ./data/hcvrp/n5000_m5_seed24610_8.npz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [31:47<00:00,  9.93s/it]
Average cost-1: 1972.2654
Average cost *8: 1507.4144
Per step inference time: 9.9322s
Total inference time: 1906.9841s
Average eval steps: 3136.35
Loading ./data/hcvrp/n5000_m7_seed24610_8.npz
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [27:27<00:00,  8.58s/it]
Average cost-1: 1735.4126
Average cost *8: 1207.5020
Per step inference time: 8.5727s
Total inference time: 1645.9597s
Average eval steps: 2704.92
Loading ./data/hcvrp/n500_m3_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [23:59<00:00,  1.41s/it]
Average cost-1: 177.0252
Average cost *8: 142.5343
Per step inference time: 1.4039s
Total inference time: 1437.5878s
Average eval steps: 457.72
Loading ./data/hcvrp/n500_m5_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [21:04<00:00,  1.23s/it]
Average cost-1: 61.0356
Average cost *8: 49.0114
Per step inference time: 1.2334s
Total inference time: 1263.0372s
Average eval steps: 399.83
Loading ./data/hcvrp/n500_m7_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [21:04<00:00,  1.23s/it]
Average cost-1: 61.0356
Average cost *8: 49.0114
Per step inference time: 1.2334s
Total inference time: 1263.0372s
Average eval steps: 399.83
Loading ./data/hcvrp/n500_m7_seed24610_8.npz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [20:16<00:00,  1.19s/it]
Average cost-1: 23.7406
Average cost *8: 22.2225
Per step inference time: 1.1869s
Total inference time: 1215.3397s
Average eval steps: 383.99
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [20:16<00:00,  1.19s/it]
Average cost-1: 23.7406
Average cost *8: 22.2225
Per step inference time: 1.1869s
Total inference time: 1215.3397s
Average eval steps: 383.99
(parco) root@I1ee26c426500201d0f:/PADR/p8/parco-main# cc
'''