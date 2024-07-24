import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import percentileofscore
from final_platoon import classify_platoons
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import arviz as az
from scipy.stats import gamma,uniform
from datetime import datetime, timedelta
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Times New Roman'
font_path = 'simhei.ttf'
font_prop = FontProperties(fname=font_path)
file_path = '/Users/pjl/PycharmProjects/tailor project/Z7_20220927_Sample_1130_1330_All_Matches.csv'
df = pd.read_csv(file_path)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df['U_Date_Time'] = pd.to_datetime(df['U_Date_Time'])
df['D_Date_Time'] = pd.to_datetime(df['D_Date_Time'])

file_path1 = '/Users/pjl/PycharmProjects/tailor project/Z6_20220927_Sample_1130_1330_All_Matches.csv'
df1 = pd.read_csv(file_path1)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df1['U_Date_Time'] = pd.to_datetime(df1['U_Date_Time'])
df1['D_Date_Time'] = pd.to_datetime(df1['D_Date_Time'])

file_path2 = '/Users/pjl/PycharmProjects/tailor project/Z5_20220927_Sample_1130_1330_All_Matches.csv'
df2 = pd.read_csv(file_path2)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df2['U_Date_Time'] = pd.to_datetime(df2['U_Date_Time'])
df2['D_Date_Time'] = pd.to_datetime(df2['D_Date_Time'])

file_path3 = '/Users/pjl/PycharmProjects/tailor project/Z4_20220927_Sample_1130_1330_All_Matches.csv'
df3 = pd.read_csv(file_path3)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df3['U_Date_Time'] = pd.to_datetime(df3['U_Date_Time'])
df3['D_Date_Time'] = pd.to_datetime(df3['D_Date_Time'])

file_path4 = '/Users/pjl/PycharmProjects/tailor project/Z3_20220927_Sample_1130_1330_All_Matches.csv'
df4 = pd.read_csv(file_path4)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df4['U_Date_Time'] = pd.to_datetime(df4['U_Date_Time'])
df4['D_Date_Time'] = pd.to_datetime(df4['D_Date_Time'])

file_path5 = '/Users/pjl/PycharmProjects/tailor project/Z2_20220927_Sample_1130_1330_All_Matches.csv'
df5 = pd.read_csv(file_path5)
# Convert 'U_Date_Time' and 'D_Date_Time' columns to Datetime type
df5['U_Date_Time'] = pd.to_datetime(df5['U_Date_Time'])
df5['D_Date_Time'] = pd.to_datetime(df5['D_Date_Time'])
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
    abs_lc_sum = df['LC'].abs().sum()
    lc1=abs_lc_sum/ave_hw
    lc2=abs_lc_sum/ave_hwind
    lc3=abs_lc_sum/ave_ind
    return  ave_hw,lc1,ave_hwind,lc2,ave_ind,lc3


# 创建一个空列表来存储结果
result_list = []

# 设置开始和结束时间
start_time = datetime.strptime("2022-09-27 11:30:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2022-09-27 13:28:00", "%Y-%m-%d %H:%M:%S")

# 设置每隔2分钟运行一次的时间间隔
interval = timedelta(minutes=2)

# 循环运行 collect_hwlc 函数，并将结果存储到列表中
current_time = start_time
while current_time <= end_time:
    # 获取当前时间范围的结果
    h1, l1, h2, l2, h3, l3 = collect_hwlc(current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          (current_time + interval).strftime("%Y-%m-%d %H:%M:%S"), df)

    # 将结果作为一行存储到列表中
    result_list.append([h1, l1, h2, l2, h3, l3])

    # 更新当前时间
    current_time += interval


# 创建一个空列表来存储结果
result_list1 = []

# 设置开始和结束时间
start_time1 = datetime.strptime("2022-09-27 11:30:00", "%Y-%m-%d %H:%M:%S")
end_time1 = datetime.strptime("2022-09-27 13:28:00", "%Y-%m-%d %H:%M:%S")

# 设置每隔2分钟运行一次的时间间隔
interval = timedelta(minutes=2)

# 循环运行 collect_hwlc 函数，并将结果存储到列表中
current_time = start_time1
while current_time <= end_time1:
    # 获取当前时间范围的结果
    h1, l1, h2, l2, h3, l3 = collect_hwlc(current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          (current_time + interval).strftime("%Y-%m-%d %H:%M:%S"), df1)

    # 将结果作为一行存储到列表中
    result_list1.append([h1, l1, h2, l2, h3, l3])

    # 更新当前时间
    current_time += interval
####################################################################
# 创建一个空列表来存储结果
result_list2 = []

# 设置开始和结束时间
start_time2 = datetime.strptime("2022-09-27 11:30:00", "%Y-%m-%d %H:%M:%S")
end_time2 = datetime.strptime("2022-09-27 13:28:00", "%Y-%m-%d %H:%M:%S")

# 设置每隔2分钟运行一次的时间间隔
interval = timedelta(minutes=2)

# 循环运行 collect_hwlc 函数，并将结果存储到列表中
current_time = start_time2
while current_time <= end_time2:
    # 获取当前时间范围的结果
    h1, l1, h2, l2, h3, l3 = collect_hwlc(current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          (current_time + interval).strftime("%Y-%m-%d %H:%M:%S"), df2)

    # 将结果作为一行存储到列表中
    result_list2.append([h1, l1, h2, l2, h3, l3])

    # 更新当前时间
    current_time += interval
# 将结果列表转换为DataFrame（如果需要）
####################################################################
# 创建一个空列表来存储结果
result_list3 = []

# 设置开始和结束时间
start_time3 = datetime.strptime("2022-09-27 11:30:00", "%Y-%m-%d %H:%M:%S")
end_time3 = datetime.strptime("2022-09-27 13:28:00", "%Y-%m-%d %H:%M:%S")

# 设置每隔2分钟运行一次的时间间隔
interval = timedelta(minutes=2)

# 循环运行 collect_hwlc 函数，并将结果存储到列表中
current_time = start_time3
while current_time <= end_time3:
    # 获取当前时间范围的结果
    h1, l1, h2, l2, h3, l3 = collect_hwlc(current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          (current_time + interval).strftime("%Y-%m-%d %H:%M:%S"), df3)

    # 将结果作为一行存储到列表中
    result_list3.append([h1, l1, h2, l2, h3, l3])

    # 更新当前时间
    current_time += interval
####################################################################
# 创建一个空列表来存储结果
result_list4 = []

# 设置开始和结束时间
start_time4 = datetime.strptime("2022-09-27 11:30:00", "%Y-%m-%d %H:%M:%S")
end_time4 = datetime.strptime("2022-09-27 13:28:00", "%Y-%m-%d %H:%M:%S")

# 设置每隔2分钟运行一次的时间间隔
interval = timedelta(minutes=2)

# 循环运行 collect_hwlc 函数，并将结果存储到列表中
current_time = start_time4
while current_time <= end_time4:
    # 获取当前时间范围的结果
    h1, l1, h2, l2, h3, l3 = collect_hwlc(current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          (current_time + interval).strftime("%Y-%m-%d %H:%M:%S"), df4)

    # 将结果作为一行存储到列表中
    result_list4.append([h1, l1, h2, l2, h3, l3])

    # 更新当前时间
    current_time += interval
####################################################################
# 创建一个空列表来存储结果
result_list5 = []

# 设置开始和结束时间
start_time5 = datetime.strptime("2022-09-27 11:30:00", "%Y-%m-%d %H:%M:%S")
end_time5 = datetime.strptime("2022-09-27 13:28:00", "%Y-%m-%d %H:%M:%S")

# 设置每隔2分钟运行一次的时间间隔
interval = timedelta(minutes=2)

# 循环运行 collect_hwlc 函数，并将结果存储到列表中
current_time = start_time5
while current_time <= end_time5:
    # 获取当前时间范围的结果
    h1, l1, h2, l2, h3, l3 = collect_hwlc(current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          (current_time + interval).strftime("%Y-%m-%d %H:%M:%S"), df5)

    # 将结果作为一行存储到列表中
    result_list5.append([h1, l1, h2, l2, h3, l3])

    # 更新当前时间
    current_time += interval


result_df = pd.DataFrame(result_list, columns=['h1', 'l1', 'h2', 'l2', 'h3', 'l3'])
result_df1 = pd.DataFrame(result_list1, columns=['h1', 'l1', 'h2', 'l2', 'h3', 'l3'])
result_df2 = pd.DataFrame(result_list2, columns=['h1', 'l1', 'h2', 'l2', 'h3', 'l3'])
result_df3 = pd.DataFrame(result_list3, columns=['h1', 'l1', 'h2', 'l2', 'h3', 'l3'])
result_df4 = pd.DataFrame(result_list4, columns=['h1', 'l1', 'h2', 'l2', 'h3', 'l3'])
result_df5 = pd.DataFrame(result_list5, columns=['h1', 'l1', 'h2', 'l2', 'h3', 'l3'])
#
# plt.scatter(  result_df['h2'],result_df['l2']/0.613,label='h1 vs l1', color='blue')
# # plt.scatter(  result_df1['h2'],result_df1['l2']/0.613,label='h1 vs l1', color='red')
#
# plt.scatter(  result_df1['h2'],result_df1['l2']/0.390,label='h1 vs l1', color='red')
# plt.scatter(  result_df2['h2'],result_df2['l2']/0.542,label='h1 vs l1', color='green')
#
# plt.scatter(  result_df3['h2'],result_df3['l2']/0.367,label='h1 vs l1', color='yellow')
# plt.scatter(  result_df4['h2'],result_df4['l2']/0.574,label='h1 vs l1', color='black')
# plt.scatter(result_df5['h2'], result_df5['l2']/0.306, label='h1 vs l1', color='pink')
# plt.show()

x=np.array(result_df['h3'])
y=np.array(result_df['l3']/0.613)

x1=np.array(result_df1['h3'])
y1=np.array(result_df1['l3']/0.390)

x2=np.array(result_df2['h3'])
y2=np.array(result_df2['l3']/0.542)

x3=np.array(result_df3['h3'])
y3=np.array(result_df3['l3']/0.367)

x4=np.array(result_df4['h3'])
y4=np.array(result_df4['l3']/0.574)

x5=np.array(result_df5['h3'])
y5=np.array(result_df5['l3']/0.306)
# Define the Bayesian model
with pm.Model() as model:
    # Priors
    p1 =pm.Gamma("p1",0.001, 0.001)
    q1 = pm.Gamma("q1",0.001, 0.001)
    p2=pm.Uniform("p2",0.5,1)
    q2=pm.Uniform("q2",0.5, 1)

    # H = pm.Normal("H",mu=5, sd=2.2)

    H = pm.Normal("H",mu=4.8, sd=0.5)
    p=p1*(H-x)**p2
    q=q1*(H-x)**q2

    # Likelihood
    likelihood = pm.Normal('L', mu=p, sd=q, observed=y)

    # MCMC sampling
    trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)
with pm.Model() as model:
    # Priors
    p1_z6 =pm.Gamma("p1_z6",0.001, 0.001)
    q1_z6 = pm.Gamma("q1_z6",0.001, 0.001)
    p2_z6=pm.Uniform("p2_z6",0.5,1)
    q2_z6=pm.Uniform("q2_z6",0.5, 1)

    # H_z6 = pm.Normal("H_z6",mu=7, sd=4)

    H_z6 = pm.Normal("H_z6",mu=4.8, sd=2)
    p_z6=p1_z6*(H_z6-x1)**p2_z6
    q_z6=q1_z6*(H_z6-x1)**q2_z6

    # Likelihood
    likelihood_z6 = pm.Normal('L_z6', mu=p_z6, sd=q_z6, observed=y1)

    # MCMC sampling
    trace_z6 = pm.sample(2000, tune=1000, cores=1, random_seed=42)
with pm.Model() as model:
    # Priors
    p1_z5 =pm.Gamma("p1_z5",0.001, 0.001)
    q1_z5 = pm.Gamma("q1_z5",0.001, 0.001)
    p2_z5=pm.Uniform("p2_z5",0.5,1)
    q2_z5=pm.Uniform("q2_z5",0.5, 1)

    # H_z5 = pm.Normal("H_z5",mu=7, sd=4)

    H_z5 = pm.Normal("H_z5",mu=4.8, sd=2)
    p_z5=p1_z5*(H_z5-x2)**p2_z5
    q_z5=q1_z5*(H_z5-x2)**q2_z5

    # Likelihood
    likelihood_z5 = pm.Normal('L_z5', mu=p_z5, sd=q_z5, observed=y2)

    # MCMC sampling
    trace_z5 = pm.sample(2000, tune=1500, cores=1, random_seed=42)

with pm.Model() as model:
    # Priors
    p1_z4 =pm.Gamma("p1_z4",0.001, 0.001)
    q1_z4 = pm.Gamma("q1_z4",0.001, 0.001)
    p2_z4=pm.Uniform("p2_z4",0.5,1)
    q2_z4=pm.Uniform("q2_z4",0.5, 1)

    # H_z4 = pm.Normal("H_z4",mu=8, sd=4)


    H_z4 = pm.Normal("H_z4",mu=4.3, sd=1.7)
    p_z4=p1_z4*(H_z4-x3)**p2_z4
    q_z4=q1_z4*(H_z4-x3)**q2_z4

    # Likelihood
    likelihood_z4 = pm.Normal('L_z4', mu=p_z4, sd=q_z4, observed=y3)

    # MCMC sampling
    trace_z4 = pm.sample(2000, tune=1000, cores=1, random_seed=42)

with pm.Model() as model:
    # Priors
    p1_z3 =pm.Gamma("p1_z3",0.001, 0.001)
    q1_z3 = pm.Gamma("q1_z3",0.001, 0.001)
    p2_z3=pm.Uniform("p2_z3",0.5,1)
    q2_z3=pm.Uniform("q2_z3",0.5, 1)

    # H_z3 = pm.Normal("H_z3",mu=8, sd=4)

    H_z3 = pm.Normal("H_z3",mu=4, sd=1)
    p_z3=p1_z3*(H_z3-x4)**p2_z3
    q_z3=q1_z3*(H_z3-x4)**q2_z3

    # Likelihood
    likelihood_z3 = pm.Normal('L_z3', mu=p_z3, sd=q_z3, observed=y4)

    # MCMC sampling
    trace_z3 = pm.sample(2000, tune=1000, cores=1, random_seed=42)

with pm.Model() as model:
    # Priors
    p1_z2 =pm.Gamma("p1_z2",0.001, 0.001)
    q1_z2 = pm.Gamma("q1_z2",0.001, 0.001)
    p2_z2=pm.Uniform("p2_z2",0.5,1)
    q2_z2=pm.Uniform("q2_z2",0.5, 1)

    # H_z2 = pm.Normal("H_z2",mu=6, sd=2)

    H_z2 = pm.Normal("H_z2",mu=4.3, sd=1.8)
    p_z2=p1_z2*(H_z2-x5)**p2_z2
    q_z2=q1_z2*(H_z2-x5)**q2_z2

    # Likelihood
    likelihood_z2 = pm.Normal('L_z2', mu=p_z2, sd=q_z2, observed=y5)

    # MCMC sampling
    trace_z2 = pm.sample(2000, tune=1000, cores=1, random_seed=42)
# sns.kdeplot(trace['H'])
# plt.show()
# print(pm.__version__)
# print(az.summary(trace))
# print(az.summary(trace_z6))

summary_trace = az.summary(trace)
summary_trace_z6 = az.summary(trace_z6)
summary_trace_z5 = az.summary(trace_z5)
summary_trace_z4 = az.summary(trace_z4)
summary_trace_z3 = az.summary(trace_z3)
summary_trace_z2 = az.summary(trace_z2)

# 将 summary 数据转换为 DataFrame
df_trace = pd.DataFrame(summary_trace)
df_trace_z6 = pd.DataFrame(summary_trace_z6)
df_trace_z5 = pd.DataFrame(summary_trace_z5)
df_trace_z4 = pd.DataFrame(summary_trace_z4)
df_trace_z3 = pd.DataFrame(summary_trace_z3)
df_trace_z2 = pd.DataFrame(summary_trace_z2)

# 将两个 DataFrame 合并
df_combined = pd.concat([df_trace, df_trace_z6, df_trace_z5, df_trace_z4, df_trace_z3, df_trace_z2], axis=1)

# 保存到 CSV 文件
# df_combined.to_csv('combined_summaryind11301230.csv', index=False)



p11 = trace['p1'].mean()
p12= trace['p2'].mean()
p21 = trace['q1'].mean()
p22= trace['q2'].mean()
p3 = trace['H'].mean()
p11_z6 = trace_z6['p1_z6'].mean()
p12_z6= trace_z6['p2_z6'].mean()
p21_z6 = trace_z6['q1_z6'].mean()
p22_z6= trace_z6['q2_z6'].mean()
p3_z6 = trace_z6['H_z6'].mean()
p11_z5 = trace_z5['p1_z5'].mean()
p12_z5= trace_z5['p2_z5'].mean()
p21_z5 = trace_z5['q1_z5'].mean()
p22_z5= trace_z5['q2_z5'].mean()
p3_z5 = trace_z5['H_z5'].mean()
p11_z4 = trace_z4['p1_z4'].mean()
p12_z4= trace_z4['p2_z4'].mean()
p21_z4 = trace_z4['q1_z4'].mean()
p22_z4= trace_z4['q2_z4'].mean()
p3_z4 = trace_z4['H_z4'].mean()
p11_z3 = trace_z3['p1_z3'].mean()
p12_z3= trace_z3['p2_z3'].mean()
p21_z3 = trace_z3['q1_z3'].mean()
p22_z3= trace_z3['q2_z3'].mean()
p3_z3 = trace_z3['H_z3'].mean()
p11_z2 = trace_z2['p1_z2'].mean()
p12_z2= trace_z2['p2_z2'].mean()
p21_z2 = trace_z2['q1_z2'].mean()
p22_z2= trace_z2['q2_z2'].mean()
p3_z2 = trace_z2['H_z2'].mean()
# 可視化
x_vis=np.linspace(1, 8, 100)
mean=p11*(p3-x_vis)**p12
std_dev= p21*(p3-x_vis)**p22
mean_z6 =p11_z6 *(p3_z6 -x_vis)**p12_z6
std_dev_z6 = p21_z6 *(p3_z6 -x_vis)**p22_z6
mean_z5 =p11_z5 *(p3_z5 -x_vis)**p12_z5
std_dev_z5 = p21_z5 *(p3_z5 -x_vis)**p22_z5
mean_z4 =p11_z4 *(p3_z4 -x_vis)**p12_z4
std_dev_z4 = p21_z4 *(p3_z4 -x_vis)**p22_z4
mean_z3 =p11_z3 *(p3_z3 -x_vis)**p12_z3
std_dev_z3 = p21_z3 *(p3_z3 -x_vis)**p22_z3
mean_z2 =p11_z2 *(p3_z2 -x_vis)**p12_z2
std_dev_z2 = p21_z2 *(p3_z2 -x_vis)**p22_z2



#
# # # 创建主坐标轴
fig1, ax1 = plt.subplots()

# 绘制数据点
ax1.plot(x, y, '*', color='red', label='匹配数据点')
ax1.plot(x1, y1, '*',color='blue',)
ax1.plot(x2, y2, '*',color='orange')

# 绘制第一个置信区间
ax1.plot(x_vis, mean, 'red', label='Z7路段')
lower_bound = mean - 1.96 * std_dev
upper_bound = mean + 1.96 * std_dev
# ax1.plot(x_vis, lower_bound, '--', color='gray')
ax1.plot(x_vis, upper_bound, '--', color='gray')

# 绘制第二个置信区间
ax1.plot(x_vis, mean_z6, 'blue', label='Z6路段')
lower_bound_z6 = mean_z6 - 1.96 * std_dev_z6
upper_bound_z6 = mean_z6 + 1.96 * std_dev_z6
# ax1.plot(x_vis, lower_bound_z6, '--', color='gray')
ax1.plot(x_vis, upper_bound_z6, '--', color='gray')

ax1.plot(x_vis, mean_z5, 'orange', label='Z5路段')
lower_bound_z5 = mean_z5 - 1.96 * std_dev_z5
upper_bound_z5 = mean_z5 + 1.96 * std_dev_z5
# ax1.plot(x_vis, lower_bound_z5, '--', color='gray')
ax1.plot(x_vis, upper_bound_z5, '--', color='gray', label='97.5%置信区间')

ax1.set_ylabel('出口路段换道系数',fontproperties=font_prop)
ax1.set_xlim([2,7])
# 添加 x 轴标签
ax1.set_xlabel('单车平均车头时距',fontproperties=font_prop)
legend1 = ax1.legend(loc='upper right',prop=font_prop)
legend1.get_frame().set_alpha(0)
# 添加次坐标轴
fig2, ax2 = plt.subplots()

# 绘制更多数据点
ax2.plot(x3, y3, '*', color='green',  label='匹配数据点')
ax2.plot(x4, y4, '*',color='pink')
ax2.plot(x5, y5, '*',color='black')

# 绘制更多置信区间
ax2.plot(x_vis, p11_z4*(p3_z4-x_vis)**p12_z4, 'green', label='Z4路段')
lower_bound_z4 = mean_z4 - 1.96 * std_dev_z4
upper_bound_z4 = mean_z4 + 1.96 * std_dev_z4
# ax2.plot(x_vis, lower_bound_z4, '--', color='gray')
ax2.plot(x_vis, upper_bound_z4, '--', color='gray')

ax2.plot(x_vis, p11_z3*(p3_z3-x_vis)**p12_z3, 'pink', label='Z3路段')
lower_bound_z3 = mean_z3 - 1.96 * std_dev_z3
upper_bound_z3 = mean_z3 + 1.96 * std_dev_z3
# ax2.plot(x_vis, lower_bound_z3, '--', color='gray')
ax2.plot(x_vis, upper_bound_z3, '--', color='gray')

ax2.plot(x_vis, p11_z2*(p3_z2-x_vis)**p12_z2, 'black', label='Z2路段')
lower_bound_z2 = mean_z2 - 1.96 * std_dev_z2
upper_bound_z2 = mean_z2 + 1.96 * std_dev_z2
# ax2.plot(x_vis, lower_bound_z2, '--', color='gray')
ax2.plot(x_vis, upper_bound_z2, '--', color='gray', label='97.5%置信区间')

ax2.set_ylabel('入口路段换道系数',fontproperties=font_prop)
# 移动次坐标轴的左侧轴线到下方
ax2.set_xlabel('单车平均车头时距',fontproperties=font_prop)

ax2.set_xlim([2,7])

legend2 = ax2.legend( loc='upper right',prop=font_prop)
# 隐藏主坐标轴的 legend 边框
legend2.get_frame().set_alpha(0)  # 隐藏次坐标轴的 legend 边框
output_path1 = f"{'/Users/pjl/Desktop/中国公路学报'}/{'inter567.png'}"
# 保存图表为高质量图片
fig1.savefig(output_path1, dpi=600, bbox_inches='tight')
output_path2 = f"{'/Users/pjl/Desktop/中国公路学报'}/{'inter234.png'}"
# 保存图表为高质量图片
fig2.savefig(output_path2, dpi=600, bbox_inches='tight')
# # output_path1 = f"{'/Users/pjl/Desktop/research_KUL/0229'}/{'ind1ff.png'}"
# # # 保存图表为高质量图片
# # fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
# # output_path2 = f"{'/Users/pjl/Desktop/research_KUL/0229'}/{'ind1cong.png'}"
# # # 保存图表为高质量图片
# # fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
# # 显示图形
plt.show()

# p1_samples = trace['H']
# p2_samples = trace_z6['H_z6']
# p3_samples = trace_z5['H_z5']
# # p4_samples = trace_z4['H_z4']
# # p5_samples = trace_z3['H_z3']
# # p6_samples = trace_z2['H_z2']
# p1_samples = trace['p1']
# p2_samples = trace_z6['p1_z6']
# p3_samples = trace_z5['p1_z5']
# p4_samples = trace_z4['p1_z4']
# p5_samples = trace_z3['p1_z3']
# p6_samples = trace_z2['p1_z2']
# # p1_samples = trace['q1']
# # p2_samples = trace_z6['q1_z6']
# # p3_samples = trace_z5['q1_z5']
# # p4_samples = trace_z4['q1_z4']
# # p5_samples = trace_z3['q1_z3']
# # p6_samples = trace_z2['q1_z2']
#
# # Plot the posterior distribution of p1
# sns.histplot(p1_samples, kde=True, stat="density", bins=30, label='Z7')
# sns.histplot(p2_samples, kde=True, stat="density", bins=5, label='Z6')
# sns.histplot(p3_samples, kde=True, stat="density", bins=30, label='Z5')
# sns.histplot(p4_samples, kde=True, stat="density", bins=30, label='Z4')
# sns.histplot(p5_samples, kde=True, stat="density", bins=30, label='Z3')
# sns.histplot(p6_samples, kde=True, stat="density", bins=30, label='Z2')
# plt.legend(frameon=False)
# plt.xlim([0,100])
# # plt.xlim([0,20])
#
# # plt.xlim([4,8])
#
# # plt.xlabel('H')
# plt.xlabel('φ')
# # plt.xlabel('φ')
#
# plt.ylabel('密度',fontproperties=font_prop)
# output_path2 = f"{'/Users/pjl/Desktop/中国公路学报'}/{'figintraφ.png'}"
# # 保存图表为高质量图片
# plt.savefig(output_path2, dpi=600, bbox_inches='tight')
# plt.show()