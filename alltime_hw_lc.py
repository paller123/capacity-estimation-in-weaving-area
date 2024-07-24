import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import percentileofscore
from final_platoon import classify_platoons
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pymc3 as pm
import arviz as az
from scipy.stats import gamma,uniform
from datetime import datetime, timedelta
plt.rcParams['font.family'] = 'Times New Roman'
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Times New Roman'
font_path = 'simhei.ttf'
font_prop = FontProperties(fname=font_path)
file_path = '/Users/pjl/PycharmProjects/tailor project/R1_NS_Z6(WA3)_20220927_1100_1500_All_Matches.csv'
df = pd.read_csv(file_path)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df['U_Date_Time'] = pd.to_datetime(df['U_Date_Time'])
df['D_Date_Time'] = pd.to_datetime(df['D_Date_Time'])


def collect_hwlc(starttime,endtime,data):
    start_time5 = starttime
    end_time5 = endtime
    df = data[(data['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])) & (
        data['D_Lane'].isin(['L1', 'L2', 'L3', 'L3r', 'L4', 'L5', 'L6'])) &
               (data['U_Date_Time'] >= start_time5) &
               (data['U_Date_Time'] <= end_time5)]
    final_platoons,unified_index_u,original_platoons,in1 = classify_platoons(df)
    HeadwayD2=[]
    HeadwayindD2=[]
    ind_hwD2=[]
    for platoon in final_platoons:
        i=platoon[0]
        HeadwayD2.append(unified_index_u[i][3])
        for j in platoon:
            if j != 0:
                HeadwayindD2.append(unified_index_u[j][3])
    for ind in in1:
        ind_hwD2.append(unified_index_u[ind[0]][3])
    ave_hw = np.mean(HeadwayD2)
    ave_hwind = np.mean(HeadwayindD2)
    ave_ind=np.mean(ind_hwD2)

    lc = df['LC'].abs().sum()

    return  ave_hw,ave_hwind,ave_ind,lc

result_listh1 = []
result_listh2 = []
result_listh3 = []
result_listlc = []
x=[]
# 设置开始和结束时间
start_time = datetime.strptime("2022-09-27 11:30:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2022-09-27 13:29:00", "%Y-%m-%d %H:%M:%S")

# 设置每隔2分钟运行一次的时间间隔
interval = timedelta(minutes=2)

# 循环运行 collect_hwlc 函数，并将结果存储到列表中
current_time = start_time
while current_time <= end_time:
    # 获取当前时间范围的结果
    h1, h2, h3, lc = collect_hwlc(current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          (current_time + interval).strftime("%Y-%m-%d %H:%M:%S"), df)

    # 将结果作为一行存储到列表中
    result_listh1.append([h1])
    result_listh2.append([h2])
    result_listh3.append([h3])
    result_listlc.append([lc])
    x.append(current_time.strftime("%H:%M"))
    # 更新当前时间
    current_time += interval

result_listh1_new=result_listh1
h1=2.5
#4
#3.7
#3.2
for i in range(1,len(result_listh1)):
    result_listh1_new[i][-1]=result_listh1_new[i-1][-1]+result_listh1[i][-1]
for i in range(1,len(result_listh1)):
    result_listh1_new[i][-1]=result_listh1_new[i][-1]-h1*i

result_listh2_new=result_listh2
h2=2.2
#2.8
#2.5
    #2.2
for i in range(1,len(result_listh2)):
    result_listh2_new[i][-1]=result_listh2_new[i-1][-1]+result_listh2[i][-1]
for i in range(1,len(result_listh2)):
    result_listh2_new[i][-1]=result_listh2_new[i][-1]-h2*i

result_listh3_new=result_listh3
h3=2.8
#2.9
#3.4
#3.6
    #3.6
for i in range(1,len(result_listh3)):
    result_listh3_new[i][-1]=result_listh3_new[i-1][-1]+result_listh3[i][-1]
for i in range(1,len(result_listh3)):
    result_listh3_new[i][-1]=result_listh3_new[i][-1]-h3*i

result_listlc_new=result_listlc
lc=124
#90/130
#70
#106
#70
for i in range(1,len(result_listlc)):
    result_listlc_new[i][-1]=result_listlc_new[i-1][-1]+result_listlc[i][-1]
for i in range(1,len(result_listlc)):
    result_listlc_new[i][-1]=result_listlc_new[i][-1]-lc*i
# Plot the data
fig, ax1 = plt.subplots(figsize=(10, 6))

# Adjust line styles, colors, and transparency
ax1.plot(x, result_listh1_new, label='车队间车头时距', color='blue', linestyle='-', linewidth=2, alpha=0.7)
ax1.plot(x, result_listh2, label='车队内车头时距', color='pink', linestyle='--', linewidth=2, alpha=0.7)
ax1.plot(x, result_listh3, label='单车车头时距', color='green', linestyle='-.', linewidth=2, alpha=0.7)

ax2 = ax1.twinx()
ax2.plot(x, result_listlc_new, label='换道次数', color='red', linestyle=':', linewidth=2, alpha=0.7)

# Customize the plot
ax1.set_xlabel('时间',fontproperties=font_prop,fontsize=20,)
ax1.set_ylabel('车头时距累积倾斜曲线(秒)', color='black',fontproperties=font_prop,fontsize=20,)
ax2.set_ylabel('换道次数累积倾斜曲线(次)', color='black',fontproperties=font_prop,fontsize=20,)

# Set background color
fig.patch.set_facecolor('#f5f5f5')  # Light gray background

# start_time = '11:56'
# end_time = '13:30'
# ax1.axvspan(start_time,end_time, facecolor='lightgray', alpha=0.5)

# start_time = '12:36'
# end_time = '13:02'
# ax1.axvspan(start_time, end_time, facecolor='lightgray', alpha=0.5)
#
# start_time = '13:06'
# end_time = '13:26'
# ax1.axvspan(start_time,end_time, facecolor='lightgray', alpha=0.5)
ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=8))

# start_time = '12:08'
# end_time = '12:14'
# ax1.axvspan(start_time, end_time, facecolor='lightgray', alpha=0.5)

# Adjust grid and legends
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left',prop=font_prop,frameon=False,fontsize=30,)
ax2.legend(loc='upper right',prop=font_prop,frameon=False,fontsize=30,)

# Title
output_path = f"{'/Users/pjl/Desktop/中国公路学报/外审修稿'}/{'fig2.png'}"
# 保存图表为高质量图片
plt.savefig(output_path, dpi=600, bbox_inches='tight')
# Show the plot
plt.show()

