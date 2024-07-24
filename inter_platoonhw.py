import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import percentileofscore
from final_platoon import classify_platoons
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
plt.rcParams['font.family'] = 'Times New Roman'
file_path = '/Users/pjl/PycharmProjects/tailor project/Z7_20220927_Sample_1130_1330_All_Matches.csv'
df = pd.read_csv(file_path)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df['U_Date_Time'] = pd.to_datetime(df['U_Date_Time'])
df['D_Date_Time'] = pd.to_datetime(df['D_Date_Time'])

# Filter data for 'U' within the specified time range and lanes
start_time1 = "2022-09-27 12:04:00"
end_time1 = "2022-09-27 12:06:00"
df_u1 = df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])) &(df['D_Lane'].isin(['L1', 'L2', 'L3','L3r', 'L4', 'L5', 'L6']))&
          (df['U_Date_Time'] >= start_time1) &
          (df['U_Date_Time'] <= end_time1)]

# Filter data for 'U' within the specified time range and lanes
start_time2 = "2022-09-27 12:14:00"
end_time2 = "2022-09-27 12:16:00"
df_u2= df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])) &(df['D_Lane'].isin(['L1', 'L2', 'L3', 'L3r','L4', 'L5', 'L6']))&
          (df['U_Date_Time'] >= start_time2) &
          (df['U_Date_Time'] <= end_time2)]

# Filter data for 'U' within the specified time range and lanes
start_time3 = "2022-09-27 12:24:00"
end_time3= "2022-09-27 12:26:00"
df_u3= df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])) &(df['D_Lane'].isin(['L1', 'L2', 'L3', 'L3r','L4', 'L5', 'L6']))&
          (df['U_Date_Time'] >= start_time3) &
          (df['U_Date_Time'] <= end_time3)]

# # Filter data for 'U' within the specified time range and lanes
start_time4 = "2022-09-27 12:34:00"
end_time4 = "2022-09-27 12:36:00"
df_u4= df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])) &(df['D_Lane'].isin(['L1', 'L2', 'L3','L3r', 'L4', 'L5', 'L6']))&
          (df['U_Date_Time'] >= start_time4) &
          (df['U_Date_Time'] <= end_time4)]

# Filter data for 'U' within the specified time range and lanes
start_time5 = "2022-09-27 12:44:00"
end_time5 = "2022-09-27 12:46:00"
df_u5= df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])) &(df['D_Lane'].isin(['L1', 'L2', 'L3', 'L3r','L4', 'L5', 'L6']))&
          (df['U_Date_Time'] >= start_time5) &
          (df['U_Date_Time'] <= end_time5)]

# start_time6 = "2022-09-27 12:43:00"
# end_time6 = "2022-09-27 12:46:00"
# df_u6= df[(df['U_Lane'].isin(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])) &(df['D_Lane'].isin(['L1', 'L2', 'L3', 'L3r','L4', 'L5', 'L6']))&
#           (df['D_Date_Time'] >= start_time6) &
#           (df['D_Date_Time'] <= end_time6)]

final_platoons1,unified_index_u1,original_platoons1,in1 = classify_platoons(df_u1)
final_platoons2,unified_index_u2,original_platoons2,in2 = classify_platoons(df_u2)
final_platoons3,unified_index_u3,original_platoons3,in3 = classify_platoons(df_u3)
final_platoons4,unified_index_u4,original_platoons4,in4 = classify_platoons(df_u4)
final_platoons5,unified_index_u5,original_platoons5,in5 = classify_platoons(df_u5)

Headway1=[]
Headwayind1=[]
Headway2=[]
Headwayind2=[]
Headway3=[]
Headwayind3=[]
Headway4=[]
Headwayind4=[]
Headway5=[]
Headwayind5=[]
platoon_speed1=[]
platoon_speed2=[]
platoon_speed3=[]
platoon_speed4=[]
platoon_speed5=[]
vehicle_length1=[]
vehicle_length2=[]
vehicle_length3=[]
vehicle_length4=[]
vehicle_length5=[]
ind_speed1=[]
ind_speed2=[]
ind_speed3=[]
ind_speed4=[]
ind_speed5=[]
ind_hw1=[]
ind_hw2=[]
ind_hw3=[]
ind_hw4=[]
ind_hw5=[]

for platoon in final_platoons1:
    i=platoon[0]
    if (unified_index_u1[i][-3]=='L5')|(unified_index_u1[i][-3]=='L4')|(unified_index_u1[i][-3]=='L3r'):
        Headway1.append(unified_index_u1[i][3])
        for j in platoon:
            if j != 0:
                Headwayind1.append(unified_index_u1[j][3])
                platoon_speed1.append(unified_index_u1[j][5])
                if unified_index_u1[j][4]>7:
                    vehicle_length1.append(unified_index_u1[j][4])
for ind in in1:
    if (unified_index_u1[ind[0]][-3] == 'L5') | (unified_index_u1[ind[0]][-3] == 'L4') | (unified_index_u1[ind[0]][-3] == 'L3r'):
        ind_speed1.append(unified_index_u1[ind[0]][5])
        ind_hw1.append(unified_index_u1[ind[0]][3])

for platoon in final_platoons2:
    i=platoon[0]
    if (unified_index_u2[i][-3]=='L5')|(unified_index_u2[i][-3]=='L4')|(unified_index_u2[i][-3]=='L3r'):
        Headway2.append(unified_index_u2[i][3])
        for j in platoon:
            if j != 0:
                Headwayind2.append(unified_index_u2[j][3])
                platoon_speed2.append(unified_index_u2[j][5])
                if unified_index_u2[j][4] > 7:
                    vehicle_length2.append(unified_index_u2[j][4])
for ind in in2:
    if (unified_index_u2[ind[0]][-3] == 'L5') | (unified_index_u2[ind[0]][-3] == 'L4') | (unified_index_u2[ind[0]][-3] == 'L3r'):
        ind_speed2.append(unified_index_u2[ind[0]][5])
        ind_hw2.append(unified_index_u2[ind[0]][3])


for platoon in final_platoons3:
    i=platoon[0]
    if (unified_index_u3[i][-3]=='L5')|(unified_index_u3[i][-3]=='L4')|(unified_index_u3[i][-3]=='L3r'):
        Headway3.append(unified_index_u3[i][3])
        for j in platoon:
            if j != 0:
                Headwayind3.append(unified_index_u3[j][3])
                platoon_speed3.append(unified_index_u3[j][5])
                if unified_index_u3[j][4] > 7:
                    vehicle_length3.append(unified_index_u3[j][4])
for ind in in3:
    if (unified_index_u3[ind[0]][-3] == 'L5') | (unified_index_u3[ind[0]][-3] == 'L4') | (unified_index_u3[ind[0]][-3] == 'L3r'):
        ind_speed3.append(unified_index_u3[ind[0]][5])
        ind_hw3.append(unified_index_u3[ind[0]][3])

for platoon in final_platoons4:
    i=platoon[0]
    if (unified_index_u4[i][-3]=='L5')|(unified_index_u4[i][-3]=='L4')|(unified_index_u4[i][-3]=='L3r'):
        Headway4.append(unified_index_u4[i][3])
        for j in platoon:
            if j != 0:
                Headwayind4.append(unified_index_u4[j][3])
                platoon_speed4.append(unified_index_u4[j][5])
                if unified_index_u4[j][4] > 7:
                    vehicle_length4.append(unified_index_u4[j][4])
for ind in in4:
    if (unified_index_u4[ind[0]][-3] == 'L5') | (unified_index_u4[ind[0]][-3] == 'L4') | (unified_index_u4[ind[0]][-3] == 'L3r'):
        ind_speed4.append(unified_index_u4[ind[0]][5])
        ind_hw4.append(unified_index_u4[ind[0]][3])

for platoon in final_platoons5:
    i=platoon[0]
    if (unified_index_u5[i][-3]=='L5')|(unified_index_u5[i][-3]=='L4')|(unified_index_u5[i][-3]=='L3r'):
        Headway5.append(unified_index_u5[i][3])
        for j in platoon:
            if j!=0:
                Headwayind5.append(unified_index_u5[j][3])
                platoon_speed5.append(unified_index_u5[j][5])
                if unified_index_u5[j][4] > 7:
                    vehicle_length5.append(unified_index_u5[j][4])
for ind in in5:
    if (unified_index_u5[ind[0]][-3] == 'L5') | (unified_index_u5[ind[0]][-3] == 'L4') | (unified_index_u5[ind[0]][-3] == 'L3r'):
        ind_speed5.append(unified_index_u5[ind[0]][5])
        ind_hw5.append(unified_index_u5[ind[0]][3])


# kde_mean1 = np.mean(Headway1)
# kde_mean2 = np.mean(Headway2)
# kde_mean3 = np.mean(Headway3)
# kde_mean4 = np.mean(Headway4)
# kde_mean5 = np.mean(Headway5)
# kde_mean1 = np.mean(Headwayind1)
# kde_mean2 = np.mean(Headwayind2)
# kde_mean3 = np.mean(Headwayind3)
# kde_mean4 = np.mean(Headwayind4)
# kde_mean5 = np.mean(Headwayind5)

# kde_mean1 = np.mean(platoon_speed1)
# kde_mean2 = np.mean(platoon_speed2)
# kde_mean3 = np.mean(platoon_speed3)
# kde_mean4 = np.mean(platoon_speed4)
# kde_mean5 = np.mean(platoon_speed5)
# kde_mean1 = np.mean(ind_hw1)
# kde_mean2 = np.mean(ind_hw2)
# kde_mean3 = np.mean(ind_hw3)
# kde_mean4 = np.mean(ind_hw4)
# kde_mean5 = np.mean(ind_hw5)
kde_mean1 = np.mean(ind_speed1)
kde_mean2 = np.mean(ind_speed2)
kde_mean3 = np.mean(ind_speed3)
kde_mean4 = np.mean(ind_speed4)
kde_mean5 = np.mean(ind_speed5)
# 在图上添加均值的垂直线
# plt.axvline(x=kde_mean1, color='r', linestyle='-.')
# plt.axvline(x=kde_mean2, color='y', linestyle='--')
# plt.axvline(x=kde_mean3, color='g', linestyle='--')
# plt.axvline(x=kde_mean4, color='b', linestyle='--')
# plt.axvline(x=kde_mean5, color='r', linestyle='--')
#
# sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Times New Roman'

colors = sns.color_palette("Set1", n_colors=5)

# # Plot KDEs with PDF
# sns.kdeplot(Headway1, fill=None, label='12:04-12:06 Q=3270 veh/h', bw_adjust=1, color=colors[0], alpha=1)
# sns.kdeplot(Headway2, fill=None, label='12:14-12:16 Q=3120 veh/h', bw_adjust=1, color=colors[1], alpha=1)
# sns.kdeplot(Headway3, fill=None, label='12:24-12:26 Q=2910 veh/h', bw_adjust=1, color=colors[2], alpha=1)
# sns.kdeplot(Headway4, fill=None, label='12:34-12:36 Q=3990 veh/h', bw_adjust=1, color=colors[3], alpha=1)
# sns.kdeplot(Headway5, fill=None, label='12:44-12:46 Q=3690 veh/h', bw_adjust=1, color=colors[4], alpha=1)
#
# # Plot KDEs with CDF
# sns.kdeplot(Headway1, fill=None, cumulative=True, bw_adjust=0.1, color=colors[0], linestyle='--',alpha=0.4)
# sns.kdeplot(Headway2, fill=None, cumulative=True, bw_adjust=0.1, color=colors[1], linestyle='--',alpha=0.4)
# sns.kdeplot(Headway3, fill=None, cumulative=True, bw_adjust=0.1, color=colors[2], linestyle='--',alpha=0.4)
# sns.kdeplot(Headway4, fill=None, cumulative=True, bw_adjust=0.1, color=colors[3], linestyle='--',alpha=0.4)
# sns.kdeplot(Headway5, fill=None, cumulative=True, bw_adjust=0.1, color=colors[4], linestyle='--',alpha=0.4)
# #
# sns.kdeplot(platoon_speed1,fill=None, label='12:04-12:06 Q=3270 veh/h',bw_adjust=1)
# sns.kdeplot(platoon_speed2,fill=None, label='12:14-12:16 Q=3120 veh/h',bw_adjust=1)
# sns.kdeplot(platoon_speed3,fill=None, label='12:24-12:26 Q=2910 veh/h',bw_adjust=1)
# sns.kdeplot(platoon_speed4,fill=None, label='12:34-12:36 Q=3990 veh/h',bw_adjust=1)
# sns.kdeplot(platoon_speed5,fill=None, label='12:44-12:46 Q=3690 veh/h',bw_adjust=1)

# sns.kdeplot(vehicle_length1,fill=None, label='12:04-12:06 Q=3270 veh/h',bw_adjust=1)
# sns.kdeplot(vehicle_length2,fill=None, label='12:14-12:16 Q=3120 veh/h',bw_adjust=1)
# sns.kdeplot(vehicle_length3,fill=None, label='12:24-12:26 Q=2910 veh/h',bw_adjust=1)
# sns.kdeplot(vehicle_length4,fill=None, label='12:34-12:36 Q=3990 veh/h',bw_adjust=1)
# sns.kdeplot(vehicle_length5,fill=None, label='12:44-12:46 Q=3690 veh/h',bw_adjust=1)

sns.kdeplot(Headwayind1,fill=None, label='12:04-12:06 Q=3270 veh/h',bw_adjust=1)
sns.kdeplot(Headwayind2,fill=None, label='12:14-12:16 Q=3120 veh/h',bw_adjust=1)
sns.kdeplot(Headwayind3,fill=None, label='12:24-12:26 Q=2910 veh/h',bw_adjust=1)
sns.kdeplot(Headwayind4,fill=None, label='12:34-12:36 Q=3990 veh/h',bw_adjust=1)
sns.kdeplot(Headwayind5,fill=None, label='12:44-12:46 Q=3690 veh/h',bw_adjust=1)
# # Plot KDEs with CDF
sns.kdeplot(Headwayind1, fill=None, cumulative=True, bw_adjust=0.1, color=colors[0], linestyle='--',alpha=0.4)
sns.kdeplot(Headwayind2, fill=None, cumulative=True, bw_adjust=0.1, color=colors[1], linestyle='--',alpha=0.4)
sns.kdeplot(Headwayind3, fill=None, cumulative=True, bw_adjust=0.1, color=colors[2], linestyle='--',alpha=0.4)
sns.kdeplot(Headwayind4, fill=None, cumulative=True, bw_adjust=0.1, color=colors[3], linestyle='--',alpha=0.4)
sns.kdeplot(Headwayind5, fill=None, cumulative=True, bw_adjust=0.1, color=colors[4], linestyle='--',alpha=0.4)
# sns.kdeplot(ind_hw1,fill=None, cumulative=True, label='12:04-12:06 Q=3270 veh/h',bw_adjust=0.1)
# sns.kdeplot(ind_hw2,fill=None, cumulative=True, label='12:14-12:16 Q=3120 veh/h',bw_adjust=0.1)
# sns.kdeplot(ind_hw3,fill=None, cumulative=True, label='12:24-12:26 Q=2910 veh/h',bw_adjust=0.1)
# sns.kdeplot(ind_hw4,fill=None, cumulative=True, label='12:34-12:36 Q=3990 veh/h',bw_adjust=0.1)
# sns.kdeplot(ind_hw5,fill=None, cumulative=True, label='12:44-12:46 Q=3690 veh/h',bw_adjust=0.1)
# plt.axvspan(3.6,3.8, alpha=0.3, color='red', label='Highlighted Region')


# sns.kdeplot(ind_speed1,fill=None, label='12:04-12:06 Q=3270 veh/h',bw_adjust=1)
# sns.kdeplot(ind_speed2,fill=None, label='12:14-12:16 Q=3120 veh/h',bw_adjust=1)
# sns.kdeplot(ind_speed3,fill=None, label='12:24-12:26 Q=2910 veh/h',bw_adjust=1)
# sns.kdeplot(ind_speed4,fill=None, label='12:34-12:36 Q=3990 veh/h',bw_adjust=1)
# sns.kdeplot(ind_speed5,fill=None, label='12:44-12:46 Q=3690 veh/h',bw_adjust=1)


plt.xlim(0,10)
# plt.xlabel("Platoon Speed(km/h)")
plt.xlabel("Inter-Platoon Headway(s)")
# plt.xlabel("Truck in Platoon(veh)")
# plt.xlabel("Individual Vehicle Headway(s)")

plt.ylabel("Density")
plt.legend(loc='upper left')
# output_path1 = f"{'/Users/pjl/Desktop/research_KUL/paper_platoon/elsarticle-template'}/{'fig8_3.png'}"
# 保存图表为高质量图片
# plt.savefig(output_path1, dpi=300, bbox_inches='tight')
plt.show()