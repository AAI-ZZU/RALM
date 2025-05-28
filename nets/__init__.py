# import numpy as np
# import matplotlib.pyplot as plt
#
# # 数据（保持不变）
# ranks = {
#     1000: {'ET': 274.86, 'AM': 466.20, '2D-Ptr': 713.22, 'DRL-li': 731.26, 'LRAM': 814.26},
#     500: {'ET': 225.54, 'AM': 253.02, '2D-Ptr': 381.46, 'DRL-li': 390.95, 'LRAM': 419.88},
#     300: {'ET': 165.72, 'AM': 159.86, '2D-Ptr': 241.98, 'DRL-li': 243.42, 'LRAM': 256.84},
#     200: {'ET': 121.76, 'AM': 111.14, '2D-Ptr': 166.30, 'DRL-li': 169.34, 'LRAM': 174.96}
# }
#
# costs = {
#     1000: {'ET': 1006, 'AM': 388.30, '2D-Ptr': 144.88, 'DRL-li': 102.45, 'LRAM': 94.21},
#     500: {'ET': 261.30, 'AM': 142.70, '2D-Ptr': 57.35, 'DRL-li': 50.80, 'LRAM': 46.68},
#     300: {'ET': 145.56, 'AM': 70.63, '2D-Ptr': 33.61, 'DRL-li': 31.95, 'LRAM': 29.33},
#     200: {'ET': 121.76, 'AM': 40.50, '2D-Ptr': 22.83, 'DRL-li': 22.31, 'LRAM': 20.50}
# }
#
# # 模型和颜色配置
# models = list(ranks[200].keys())
# colors = ['#E2F1D5', '#D0E4F0', '#F8C0C0', '#FAE8B0', '#6B5B95']  # 使用更浅的颜色
# patterns = ['/', '\\', 'x', '+', '|']
#
#
# # X 轴位置
# x_values = np.arange(len(models))  # 定义 x_values
#
# # 创建子图
# fig, axs = plt.subplots(2, 4, figsize=(16, 10))  # 4列2行布局
# # 绘制图表
# for idx, scale in enumerate(sorted(ranks.keys())):
#     # ------ 数据处理 ------
#     rank_values = [ranks[scale][model] / scale for model in models]
#     cost_values = [costs[scale][model] for model in models]
#
#     # ------ 绘制折线图（Cost）------
#     ax_cost = axs[0, idx]  # 上方的 Cost 图
#     ax_cost.plot(
#         x_values, cost_values,
#         linestyle='--',  # 使用虚线
#         marker='o',  # 圆圈标记
#         markersize=8,
#         linewidth=2,
#         color='blue',
#         zorder=100,
#         markeredgecolor='black',
#         markerfacecolor='blue'
#     )
#
#     # 标注
#     for i, value in enumerate(cost_values):
#         ax_cost.text(x_values[i], value + 15,
#                      f'{value:.1f}', fontsize=10, ha='center', color='blue', zorder=200)
#
#     ax_cost.set_title(f'Scale: {scale}', fontsize=14, fontweight='bold')
#     ax_cost.set_xticks(x_values)
#     ax_cost.set_xticklabels(models, fontsize=12)
#     ax_cost.set_ylabel('Cost', fontsize=12, color='blue')
#
#     # 设置Y轴范围和刻度
#     if scale == 1000:
#         ax_cost.set_ylim(0, 1100)  # 设置范围
#         ax_cost.set_yticks(np.arange(0, 1100, 200))  # 设置间隔为200的刻度
#     elif scale == 500:
#         ax_cost.set_ylim(0, 300)
#         ax_cost.set_yticks(np.arange(0, 300, 50))
#     elif scale == 300:
#         ax_cost.set_ylim(0, 160)
#         ax_cost.set_yticks(np.arange(0, 160, 20))
#     elif scale == 200:
#         ax_cost.set_ylim(0, 200)
#         ax_cost.set_yticks(np.arange(0, 200, 25))
#
#     ax_cost.grid(True, linestyle=':')
#
#     # ------ 绘制柱状图（Rank）------
#     ax_rank = axs[1, idx]  # 下方的 Rank 图
#     bars = ax_rank.bar(
#         x_values, rank_values,
#         color=colors, alpha=0.5,
#         width=0.5,
#         zorder=1,
#         edgecolor='black',
#         linewidth=0.5
#     )
#
#     # 添加柱状图纹理
#     for j, bar in enumerate(bars):
#         bar.set_hatch(patterns[j % len(patterns)])
#
#         # 柱状图标注
#     for i, value in enumerate(rank_values):
#         ax_rank.text(x_values[i], value + 0.02,
#                      f'{value:.2f}', fontsize=10, ha='center', color='black')
#
#     ax_rank.set_xticks(x_values)
#     ax_rank.set_xticklabels(models, fontsize=12)
#     ax_rank.set_ylabel('Rank/N', fontsize=12)
#
#     # 设置Y轴范围
#     ax_rank.set_ylim(0, max(rank_values) * 1.15)
#     ax_rank.grid(True, linestyle=':')
#
# # 调整布局
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
# # # 添加合适的图例
# # handles = [plt.Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Cost'),
# #            plt.Rectangle((0, 0), 1, 1, color='#D0E4F0', alpha=0.5, hatch='/', label='Model 1'),
# #            plt.Rectangle((0, 0), 1, 1, color='#E2F1D5', alpha=0.5, hatch='\\', label='Model 2'),
# #            plt.Rectangle((0, 0), 1, 1, color='#F8C0C0', alpha=0.5, hatch='x', label='Model 3'),
# #            plt.Rectangle((0, 0), 1, 1, color='#FAE8B0', alpha=0.5, hatch='+', label='Model 4'),
# #         plt.Rectangle((0, 0), 1, 1, color='#FAE8B0', alpha=0.5, hatch='+', label='Model 4')
# #            ]
#
# # fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.9, 0.9), fontsize=10)
#
# # 保存为PDF
# plt.savefig('cost_rank_comparison_full.pdf', format='pdf', bbox_inches='tight', dpi=4000)
#
# plt.show()