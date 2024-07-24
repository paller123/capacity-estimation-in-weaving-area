# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.lines import Line2D
# import datetime
# from datetime import datetime, timedelta
# import matplotlib.dates as mdates
# import seaborn as sns
# # 读取文件数据
# plt.rcParams['font.family'] = 'Times New Roman'
#
# file_path = '/Users/pjl/PycharmProjects/tailor project/Z3_20220927_Sample_1130_1330_All_Matches.csv'
# df= pd.read_csv(file_path)
# # 将 'U_Date_Time' 列转换为 Datetime 类型
# df['U_Date_Time'] = pd.to_datetime(df['U_Date_Time'])
# # 过滤数据，仅保留在指定时间范围内的数据
# start_time = "2022-09-27 12:25:00"
# end_time = "2022-09-27 12:26:00"
# df = df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5'])) &
#                    (df['U_Date_Time'] >= start_time) &
#                    (df['U_Date_Time'] <= end_time)]
# # 按照时间对数据进行排序
# df = df.sort_values(by='U_Date_Time')
#
# # 创建图表
# fig, ax = plt.subplots(figsize=(12, 6))
#
# # 获取车道数量
# lanes = sorted(df['U_Lane'].unique(), key=lambda x: int(x[1:]))  # 按照L1到L5的顺序排列
#
#
#
#
#
# # 绘制每辆车的矩形
# # 绘制每辆车的散点图
# for lane_idx, lane in enumerate(df['U_Lane'].unique()):
#     lane_data = df[df['U_Lane'] == lane]
#
#     # 根据每辆车的VL_Ave值来设置点的大小
#     rect_width = [50 if vl_ave < 7 else 200 for vl_ave in lane_data['VL_Ave']]
#
#     sc = ax.scatter(lane_data['U_Date_Time'], [lane_idx + 1] * len(lane_data), c=lane_data['U_Speed'], cmap='summer', s=rect_width)
#     # Annotate each point with its index
#     # for i, txt in enumerate(lane_data.index):
#     #     ax.annotate(txt, (lane_data['U_Date_Time'].iloc[i], lane_idx + 1), textcoords="offset points", xytext=(0, -5), ha='center', fontsize=8)
#
# # 创建额外的轴用于颜色条
# cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
# cbar = plt.colorbar(sc, cax=cax)
# cbar.set_label('U_Speed')
# # 设置横轴和纵轴标签
# ax.set_xlabel('Time')
# ax.set_ylabel('Lanes')
#
# # 设置纵轴刻度
# ax.set_yticks(list(range(1, 6)))  # 在两端各添加一个空白刻度
# ax.set_yticklabels( lanes )  # 不显示实际标签
#
# plt.show()

#
# import pandas as pd
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Times New Roman'
#
# file_path = '/Users/pjl/PycharmProjects/tailor project/Z3_20220927_Sample_1130_1330_All_Matches.csv'
# df = pd.read_csv(file_path)
#
# # Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
# df['U_Date_Time'] = pd.to_datetime(df['U_Date_Time'])
# df['D_Date_Time'] = pd.to_datetime(df['D_Date_Time'])
#
# # Filter data for 'U' within the specified time range and lanes
# start_time = "2022-09-27 12:25:00"
# end_time = "2022-09-27 12:26:00"
# df_u = df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5'])) &
#           (df['U_Date_Time'] >= start_time) &
#           (df['U_Date_Time'] <= end_time)]
#
# # Filter data for 'D' within the specified time range and lanes
# df_d = df[(df['D_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5'])) &
#           (df['D_Date_Time'] >= start_time) &
#           (df['D_Date_Time'] <= end_time)]
#
# # Sort data by time
# df_u = df_u.sort_values(by='U_Date_Time')
# df_d = df_d.sort_values(by='D_Date_Time')
#
# # Create the plot
# fig, ax = plt.subplots(figsize=(12, 6))
#
# # Get lane information
# lanes = sorted(df_u['U_Lane'].unique(), key=lambda x: int(x[1:]))
#
# # Plot 'U' scatter points with labels
# for lane_idx, lane in enumerate(df_u['U_Lane'].unique()):
#     lane_data = df_u[df_u['U_Lane'] == lane]
#     rect_width = [50 if vl_ave < 7 else 200 for vl_ave in lane_data['VL_Ave']]
#     sc = ax.scatter(lane_data['U_Date_Time'], [lane_idx + 1] * len(lane_data), c=lane_data['U_Speed'], cmap='summer', s=rect_width)
#     for i, txt in enumerate(lane_data.index):
#         ax.annotate(txt, (lane_data['U_Date_Time'].iloc[i], lane_idx + 1), textcoords="offset points", xytext=(0, -5), ha='center', fontsize=8)
# # Plot 'D' scatter points with labels on the same set of axes, adjusted y-axis position
# for lane_idx, lane in enumerate(df_d['D_Lane'].unique()):
#     lane_data = df_d[df_d['D_Lane'] == lane]
#     rect_width = [50 if vl_ave < 7 else 200 for vl_ave in lane_data['VL_Ave']]
#     sc = ax.scatter(lane_data['D_Date_Time'], [lane_idx + 1 + len(df_u['U_Lane'].unique())] * len(lane_data), c=lane_data['D_Speed'], cmap='autumn', s=rect_width)
#     for i, txt in enumerate(lane_data.index):
#         ax.annotate(txt, (lane_data['D_Date_Time'].iloc[i], lane_idx + 1+ len(df_u['U_Lane'].unique())), textcoords="offset points", xytext=(0, -5), ha='center', fontsize=8)
# # Create a colorbar
# cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
# cbar = plt.colorbar(sc, cax=cax)
# cbar.set_label('U_Speed / D_Speed')
#
# # Set axis labels
# ax.set_xlabel('Time')
# ax.set_ylabel('Lanes (U/D)', color='green')
#
# # Set y-axis ticks and labels for both 'U' and 'D'
# combined_lanes = lanes + [f"{lane} (D)" for lane in lanes]
# ax.set_yticks(list(range(1, 2 * len(lanes) + 1)))
# ax.set_yticklabels(combined_lanes)
#
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Times New Roman'
font_path = 'simhei.ttf'
font_prop = FontProperties(fname=font_path)
file_path = '/Users/pjl/PycharmProjects/tailor project/Z3_20220927_Sample_1130_1330_All_Matches.csv'
df = pd.read_csv(file_path)

# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df['U_Date_Time'] = pd.to_datetime(df['U_Date_Time'])
df['D_Date_Time'] = pd.to_datetime(df['D_Date_Time'])

# Filter data for 'U' within the specified time range and lanes
start_time = "2022-09-27 12:29:00"
end_time = "2022-09-27 12:30:00"
df_u = df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5','L6'])) &(df['D_Lane'].isin(['L1', 'L2', 'L3',  'L4','L5','L6']))&
          (df['U_Date_Time'] >= start_time) &
          (df['U_Date_Time'] <= end_time)]

# Sort 'U' data by time
df_u = df_u.sort_values(by='U_Date_Time')

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Get lane information
lanes = sorted(df_u['U_Lane'].unique(), key=lambda x: int(x[1:]))
# lanes1 = sorted(df_u['D_Lane'].unique(), key=lambda x: int(x[1:]))

# # Plot 'U' scatter points with labels
# unified_index_u = {}  # Mapping for 'U' series indices
# for lane_idx, lane in enumerate(df_u['U_Lane'].unique()):
#     lane_data_u = df_u[df_u['U_Lane'] == lane]
#     rect_width = [50 if vl_ave < 7 else 200 for vl_ave in lane_data_u['VL_Ave']]
#     sc = ax.scatter(lane_data_u['U_Date_Time'], [lane_idx + 1] * len(lane_data_u), c=lane_data_u['U_Speed'], cmap='RdYlGn', s=rect_width, vmin=0, vmax=120)
#     for i, txt in enumerate(lane_data_u.index):
#         unified_index_u[txt] = (lane_data_u['U_Date_Time'].iloc[i], lane_idx + 1)
#         ax.annotate(txt, (lane_data_u['U_Date_Time'].iloc[i], lane_idx + 1), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=5)
#
# # Plot 'D' scatter points with labels on the same set of axes, adjusted y-axis position
# for i, txt_u in enumerate(df_u.index):
#     lane_data_d = df[(df['D_Lane'] == df_u['D_Lane'].iloc[i]) & (df['D_Date_Time'] == df_u['D_Date_Time'].iloc[i])]
#     if not lane_data_d.empty:
#         lane_idx_d = lane_data_d['D_Lane'].astype(str).str.extract('(\d+)').astype(int).iloc[0]
#         rect_width_d = [50 if vl_ave < 7 else 200 for vl_ave in lane_data_d['VL_Ave']]
#         sc = ax.scatter(lane_data_d['D_Date_Time'], [lane_idx_d  + len(df_u['U_Lane'].unique())] * len(lane_data_d), c=lane_data_d['D_Speed'], cmap='RdYlGn', s=rect_width_d, vmin=0, vmax=120)
#         for j, txt_d in enumerate(lane_data_d.index):
#             unified_index_d = (lane_data_d['D_Date_Time'].iloc[j], lane_idx_d + len(df_u['U_Lane'].unique()))
#             ax.annotate(txt_d, unified_index_d, textcoords="offset points", xytext=(0, -10), ha='center', fontsize=5)

# Plot 'U' scatter points with labels
unified_index_u = {}  # Mapping for 'U' series indices
global_counter = 0  # Global counter for overall numbering
for lane_idx, lane in enumerate(df_u['U_Lane'].unique(), start=1):
    lane_data_u = df_u[df_u['U_Lane'] == lane]
    rect_width = [50 if vl_ave < 7 else 200 for vl_ave in lane_data_u['VL_Ave']]
    y_position = lane_idx
    sc = ax.scatter(lane_data_u['U_Date_Time'], [y_position] * len(lane_data_u), c=lane_data_u['U_Speed'], cmap='RdYlGn', s=rect_width, vmin=0, vmax=120)
    for i, txt in enumerate(range(1, len(lane_data_u) + 1), start=1):
        global_counter += 1
        unified_index_u[lane_data_u.index[i - 1]] = (lane_data_u['U_Date_Time'].iloc[i - 1], y_position, global_counter)
        ax.annotate(global_counter, (lane_data_u['U_Date_Time'].iloc[i - 1], y_position), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=5)

# Plot 'D' scatter points with labels on the same set of axes, adjusted y-axis position
for i, txt_u in enumerate(df_u.index):
    lane_data_d = df[(df['D_Lane'] == df_u['D_Lane'].iloc[i]) & (df['D_Date_Time'] == df_u['D_Date_Time'].iloc[i])]
    if not lane_data_d.empty:
        lane_idx_d = lane_data_d['D_Lane'].astype(str).str.extract('(\d+)').astype(int).iloc[0]
        rect_width_d = [50 if vl_ave < 7 else 200 for vl_ave in lane_data_d['VL_Ave']]
        y_position_d = lane_idx_d + len(df_u['U_Lane'].unique())
        for j, txt_d in enumerate(range(1, len(lane_data_d) + 1), start=1):
            unified_index_d = (lane_data_d['D_Date_Time'].iloc[j - 1], y_position_d, unified_index_u[txt_u][2])
            print(lane_data_d['D_Date_Time'].iloc[j - 1])
            # print([y_position_d] * len(lane_data_d))
            ax.scatter(lane_data_d['D_Date_Time'].iloc[j - 1], [y_position_d] * len(lane_data_d), c=lane_data_d['D_Speed'], cmap='RdYlGn', s=rect_width_d, vmin=0, vmax=120)
            ax.annotate(str(unified_index_u[txt_u][2]), (unified_index_d[0], unified_index_d[1]),
                        textcoords="offset points", xytext=(0, -10), ha='center', fontsize=5)

cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cbar = plt.colorbar(sc, cax=cax)

cbar.set_label('车辆速度(km/h)',fontproperties=font_prop)

# Set axis labels
ax.set_xlabel('时间',fontproperties=font_prop)
ax.set_ylabel('车道', color='green',fontproperties=font_prop)

# Set y-axis ticks and labels for 'U'
combined_lanes = lanes + [f"{lane} (D)" for lane in lanes if "L6" not in lane]

# combined_lanes = lanes + [f"{lane} (D)" for lane in lanes if "L6" not in lane]
ax.set_yticks(list(range(1, 2 * len(lanes))))
ax.set_yticklabels(combined_lanes)
output_path1 = f"{'/Users/pjl/Desktop/中国公路学报'}/{'fig5.png'}"
# 保存图表为高质量图片
plt.savefig(output_path1, dpi=600, bbox_inches='tight')
plt.show()

