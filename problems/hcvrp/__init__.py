# import pandas as pd
#
# # 假设我们从一个文件中读取数据，这里用字典模拟数据
#
# data = pd.read_csv("RANS.csv")
#
#
#
# # data = {
# #     "Instance": ["eil51", "berlin52", "st70", "pr76", "eil76", "rat99", "rd100", "KroA100", "KroB100", "KroC100", "KroD100"],
# #     "Opt.": [426, 7542, 675, 108159, 538, 1211, 7910, 21282, 22141, 20749, 21294],
# #     "POMO Cost": [429, 7545, 677, 108681, 544, 1270, 7912, 21486, 22285, 20755, 21488],
# #     "POMO Gap": [0.82, 0.04, 0.31, 0.48, 1.18, 4.90, 0.03, 0.96, 0.65, 0.30, 0.91],
# #     "Sym-NCO Cost": [432, 7544, 677, 108388, 544, 1261, 7911, 21397, 22378, 20760, 21696],
# #     "Sym-NCO Gap": [1.39, 0.03, 0.31, 0.21, 1.18, 4.17, 0.02, 0.54, 1.07, 0.10, 1.89]
# # }
#
# # 将数据转化为 DataFrame
# df = pd.DataFrame(data)
#
# # 生成 LaTeX 表格
# latex_table = df.to_latex(index=False, float_format="%.2f", caption="Performance comparison in real-world instances in TSPLIB.", label="tab:performance_comparison")
#
# # 打印latex表格
# print(latex_table)