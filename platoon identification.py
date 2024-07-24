import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from scipy.stats import percentileofscore
plt.rcParams['font.family'] = 'Times New Roman'

# 读取数据
data = pd.read_csv('E313tR1_ES_Ramp_Z3(WA1r)_20220927_1100_1500_All_Matches.csv')
data1 = pd.read_csv('R1_NS_Z4(WA1)_20220927_1100_1500_All_Matches.csv')
data1=pd.concat((data,data1))# 替换成您的数据文件路径
data2 = pd.read_csv('R1_NS_Z5(WA2)_20220927_1100_1500_All_Matches.csv')  # 替换成您的数据文件路径
data3 = pd.read_csv('R1_NS_Z6(WA3)_20220927_1100_1500_All_Matches.csv')  # 替换成您的数据文件路径
data4= pd.read_csv('R1_NS_Z7(WA4)_20220927_1100_1500_All_Matches.csv')  # 替换成您的数据文件路径
data5 = pd.read_csv('R1_NS_Z8(WA5)_20220927_1100_1500_All_Matches.csv')  # 替换成您的数据文件路径
data6 = pd.read_csv('R1_NS_Z9(WA6)_20220927_1100_1500_All_Matches.csv')  # 替换成您的数据文件路径
data7 = pd.read_csv('R1_NS_Z10(WA7)_20220927_1100_1500_All_Matches.csv') # 替换成您的数据文件路径

################new data######################
data1=data1[((data1['D_Lane'] == 'L4') | (data1['D_Lane'] == 'L5'))]
data2=data2[((data2['D_Lane'] == 'L4') | (data2['D_Lane'] == 'L5'))]
data3=data3[((data3['D_Lane'] == 'L4') | (data3['D_Lane'] == 'L5'))]
data4=data4[((data4['D_Lane'] == 'L4') | (data4['D_Lane'] == 'L5'))]
data5=data5[((data5['D_Lane'] == 'L4') | (data5['D_Lane'] == 'L5'))]
data6=data6[((data6['D_Lane'] == 'L4') | (data6['D_Lane'] == 'L5'))]
data7=data7[((data7['D_Lane'] == 'L3r') | (data7['D_Lane'] == 'L4') |(data7['D_Lane'] == 'L5'))]

# 替换成您的数据文件路径
# 将日期时间列转换为 Pandas 的日期时间格式
# data['D_Date_Time'] = pd.to_datetime(data['D_Date_Time'])
# data1['D_Date_Time'] = pd.to_datetime(data1['D_Date_Time'])
# data2['D_Date_Time'] = pd.to_datetime(data2['D_Date_Time'])
# data3['D_Date_Time'] = pd.to_datetime(data3['D_Date_Time'])
# data4['D_Date_Time'] = pd.to_datetime(data4['D_Date_Time'])
# data5['D_Date_Time'] = pd.to_datetime(data5['D_Date_Time'])
# data6['D_Date_Time'] = pd.to_datetime(data6['D_Date_Time'])
# data7['D_Date_Time'] = pd.to_datetime(data7['D_Date_Time'])

# # 设置日期时间列为数据框的索引
# # data.set_index('D_Date_Time', inplace=True)
# data1.set_index('D_Date_Time', inplace=True)
# data2.set_index('D_Date_Time', inplace=True)
# data3.set_index('D_Date_Time', inplace=True)
# data4.set_index('D_Date_Time', inplace=True)
# data5.set_index('D_Date_Time', inplace=True)
# data6.set_index('D_Date_Time', inplace=True)
# data7.set_index('D_Date_Time', inplace=True)
def information_extact_time1(data):
# data3=data3[(data3['VL_Ave']<7)]
    data.sort_values(by='D_Date_Time', inplace=True)
    data_select=data
    # 创建一个空列表来存储车队数据
    convoy_data = []
    individual_data=[]
    # 遍历不同的U_Lane值
    for lane in data_select['D_Lane'].unique():
        # 为当前车道筛选数据
        lane_data = data_select[(data_select['D_Lane']== lane)].sort_values(by='D_Date_Time')
        # 初始化用于跟踪车队数据的变量
        current_convoy = []
        current_convoy_start_headway = None
        current_intraplatoon=[]
        current_speed=[]
        # 遍历车道数据的行
        for index, row in lane_data.iterrows():
            # 检查U_Headway是否小于2.5
            if row['D_Headway'] < 4:
                # 如果current_convoy为空，更新起始车头时距
                # 添加前一行的数据到车队数据
                prev_row_index = lane_data.index.get_loc(index) - 1
                prev_row = lane_data.iloc[prev_row_index].to_dict() if prev_row_index >= 0 else None
                current_intraplatoon.append(row['D_Headway'])
                current_speed.append(row['D_Speed'])
                if prev_row:
                    if prev_row['D_Headway'] >4:
                            current_convoy.append(prev_row)
                            if current_convoy:
                                current_convoy_start_headway = prev_row['D_Headway']
                # 将行追加到current_convoy
                current_convoy.append(row.to_dict())
            else:
                # 检查current_convoy是否有数据
                if current_convoy:
                    # 将车队数据和起始车头时距追加到convoy_data
                    convoy_data.append({
                        '车道': lane,
                        '车队': current_convoy,
                        '起始车头时距': current_convoy_start_headway,
                        "队内车头时距": current_intraplatoon,
                        '队内速度': current_speed
                    })
                    # 重置current_convoy和起始车头时距
                    current_convoy = []
                    current_convoy_start_headway = None
                    current_intraplatoon = []
                    current_speed=[]
                else:
                    individual_data.append({
                        '车道': lane,
                        '车辆': row,
                        "车头时距":row['D_Headway'],
                        '车速': row['D_Speed']
                    })
        # 检查是否在最后一行之后还有剩余的车队数据
        if current_convoy:
            convoy_data.append({
                '车道': lane,
                '车队': current_convoy,
                '起始车头时距': current_convoy_start_headway,
                "队内车头时距":current_intraplatoon,
                '队内速度': current_speed
            })
    # np.savetxt('zone4_platoon.csv', convoy_data, delimiter=",")
    # # 打印或根据需要使用convoy_data
    # for convoy in convoy_data:


    ################################5分钟分类##########################
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime, timedelta

    # 假设 convoy_data 是一个具有你提供的结构的字典列表
    # 如果不是，请根据实际的数据结构进行调整

    # 提取每个车队的第一辆车的 'U_Date_Time'

    first_vehicle_times = pd.to_datetime([convoy['车队'][0]['D_Date_Time'].split('.')[0] for convoy in convoy_data],
                                         errors='coerce')
    individual_vehicle_times = pd.to_datetime([vehicle['车辆']['D_Date_Time'].split('.')[0] for vehicle in individual_data],
                                         errors='coerce')

    # 设置时间间隔（例如，5分钟）
    interval = timedelta(minutes=10)

    # 定义起始时间和结束时间
    start_time = datetime.strptime('11:30:00', '%H:%M:%S')
    end_time = datetime.strptime('13:30:00', '%H:%M:%S')
    # 定义特定日期
    specific_date = datetime.strptime('2022-09-27', '%Y-%m-%d').date()
    start_time = datetime.combine(specific_date, start_time.time())
    end_time = datetime.combine(specific_date, end_time.time())
    # 创建一个空的字典，用于存储每个时间段的 DataFrame
    interval_data_dict = {}
    interval_individual_data_dict={}
    # 遍历每个时间段
    current_time = start_time
    while current_time < end_time:
        end_interval_time = current_time + interval


        # 从 convoy_data 中筛选在当前时间段内的所有车队数据
        selected_convoy_data = [convoy for convoy, time in zip(convoy_data, first_vehicle_times) if current_time <= time < end_interval_time]
        selected_individual_data = [vehicle for vehicle, time in zip(individual_data, individual_vehicle_times) if current_time <= time < end_interval_time]

        # 将选定的车队数据存储到 interval_data_dict 中
        if selected_convoy_data:
            interval_data_dict[current_time] = pd.DataFrame(selected_convoy_data)
        if selected_individual_data:
            interval_individual_data_dict[current_time] = pd.DataFrame(selected_individual_data)
        current_time = end_interval_time

    # 示例：访问 11:30-11:35 时间段的数据
    # if start_time in interval_data_dict:
        # print("Data for 11:30-11:35:")
        # print(interval_data_dict[start_time])
    test_time = datetime.strptime('11:30:00', '%H:%M:%S')
    # test_time1 = datetime.strptime('11:30:00', '%H:%M:%S')
    # test_time2 = datetime.strptime('11:30:00', '%H:%M:%S')
    # test_time3 = datetime.strptime('11:30:00', '%H:%M:%S')
    # test_time4 = datetime.strptime('11:30:00', '%H:%M:%S')

    test_time = datetime.combine(specific_date, test_time.time())
    # test_time1 = datetime.strptime('12:18:00', '%H:%M:%S')
    # test_time1 = datetime.combine(specific_date, test_time1.time())
    # test_time2 = datetime.combine(specific_date, test_time2.time())
    # test_time3 = datetime.combine(specific_date, test_time3.time())
    # test_time4 = datetime.combine(specific_date, test_time4.time())

    # filtered_data = interval_data_dict[test_time]
    keys_to_get = [test_time]

    # 存储多个键对应的值
    filtered_data = []
    filtered_individual_data=[]
    # 逐个获取每个键对应的值
    for key in keys_to_get:
        if key in interval_data_dict:
            filtered_data.append(interval_data_dict[key])
            filtered_individual_data.append(interval_individual_data_dict[key])

     # filtered_individual_data= interval_individual_data_dict[test_time]

    # filtered_data1 = interval_data_dict[test_time][(interval_data_dict[test_time]['车道'] == 'L5')]
    # filtered_data2 = interval_data_dict[test_time1][interval_data_dict[test_time1]['车道'] == 'L4']

    # 示例：绘制 11:30-11:35 时间段的直方图
    # plt.hist(pd.to_datetime(filtered_data['车队'].apply(lambda x: x[0]['U_Date_Time'].split('.')[0])),
    #          bins=20, alpha=0.5, label='第一辆车时间')
    # plt.xlabel('时间')
    # plt.ylabel('频率')
    # plt.title('12:08-11:10 时间段的第一辆车时间分布')
    # plt.show()
    ###########################################################
    # 统计车队中车辆数目和车辆平均速度
    # frequency_data =  []
    # frequency_data_truck =  []
    # frequency_data_speed =  []
    #
    # for index, row in filtered_data.iterrows():
    #     row_length = len(row['车队'])
    #     frequency_data.append(row_length)
    #     i=0
    #     totalspeed=0
    #     for index in row['车队']:
    #         if index['VL_Ave']>7:
    #             i=i+1
    #     frequency_data_truck.append(i)
    #     for index in row['车队']:
    #         totalspeed+=index['U_Speed']
    #         i=i+1
    #     frequency_data_speed.append(totalspeed/i)
    #
    # # 统计车队中车辆数目和车辆平均速度
    # frequency_data1 =  []
    # frequency_data_truck1 =  []
    # frequency_data_speed1 =  []
    #
    # for index, row in filtered_data1.iterrows():
    #     row_length = len(row['车队'])
    #     frequency_data1.append(row_length)
    #     i=0
    #     totalspeed=0
    #     for index in row['车队']:
    #         if index['VL_Ave']>7:
    #             i=i+1
    #     frequency_data_truck1.append(i)
    #     for index in row['车队']:
    #         totalspeed+=index['U_Speed']
    #         i=i+1
    #     frequency_data_speed1.append(totalspeed/i)
    # #
    # # 统计同一车道不同时间段车队中车辆数目和车辆平均速度
    # frequency_data2 =  []
    # frequency_data_truck2 =  []
    # frequency_data_speed2 =  []
    #
    # for index, row in filtered_data2.iterrows():
    #     row_length = len(row['车队'])
    #     frequency_data2.append(row_length)
    #     i=0
    #     totalspeed=0
    #     for index in row['车队']:
    #         if index['VL_Ave']>7:
    #             i=i+1
    #     frequency_data_truck2.append(i)
    #     for index in row['车队']:
    #         totalspeed+=index['U_Speed']
    #         i=i+1
    #     frequency_data_speed2.append(totalspeed/i)
    # # plt.hist([frequency_data, frequency_data_truck], bins=range(1, max(max(frequency_data), max(frequency_data_truck)) + 2), alpha=0.7, label=['所有数据', 'VL_Ave > 7 的数据'])
    # ###########KDE###################
    # combined_list = list(zip(frequency_data, frequency_data_speed))
    # combined_list1 = list(zip(frequency_data1, frequency_data_speed1))
    #
    # combined_list2 = list(zip(frequency_data, filtered_data['起始车头时距']))
    # combined_list3 = list(zip(frequency_data1, filtered_data1['起始车头时距']))
    #
    # combined_list4 = list(zip(filtered_data['起始车头时距'],frequency_data_speed))
    # combined_list5 = list(zip( filtered_data1['起始车头时距'],frequency_data_speed1))
    #
    # combined_list6 = list(zip(frequency_data_truck,frequency_data_speed))
    # combined_list7 = list(zip(frequency_data_truck1,frequency_data_speed1))
    #
    # combined_list8 = list(zip(frequency_data_truck,filtered_data['起始车头时距']))
    # combined_list9 = list(zip(frequency_data_truck1,filtered_data1['起始车头时距']))
    #
    # combined_list10 = list(zip(frequency_data, frequency_data_speed))
    # combined_list11= list(zip(frequency_data2, frequency_data_speed2))
    #
    # combined_list12 = list(zip(frequency_data, filtered_data['起始车头时距']))
    # combined_list13= list(zip(frequency_data2, filtered_data2['起始车头时距']))
    #
    #
    # subtracted_list = [(x1 - x2, y1 - y2) for x1, y1 in combined_list2 for x2, y2 in combined_list3]
    # x_values, y_values = zip(*subtracted_list)
    #
    # # 使用 seaborn 绘制 KDE 图
    # sns.kdeplot(x=x_values,y=y_values,  fill=True)
    # # 添加 x=0 和 y=0 的线
    # plt.axvline(0, color='purple', linestyle='dashed', linewidth=0.5)
    # plt.axhline(0, color='purple', linestyle='dashed', linewidth=0.5)
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # plt.yticks(range(-10, 11,2))
    #
    # plt.xticks(range(-10, 11,2))
    # # 添加标题和标签
    # plt.xlabel("Delta platoon size(veh)")
    # # plt.ylabel("Delta platoon speed(km/h)")
    # plt.ylabel("Delta inter-platoon headway(s)")
    #
    #
    # # 显示图形
    # plt.show()


    ###########KDE###################
    # # 计算频率直方图
    # hist, bins, _ = plt.hist(frequency_data_speed, bins=10, alpha=0.7, color='lightblue')
    #
    # plt.xlabel('Platoon Average Speed (Km/h)')
    # plt.ylabel('Frequency')
    #
    # # 设置纵坐标刻度只显示整数值
    # plt.yticks(range(int(min(hist)), int(max(hist)) + 1))
    # # plt.legend()
    # plt.show()


    ##########################绘制PDF################
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(10, 6))
    # # Assuming 'U_Date_Time' is the column you want to frequency_dataplot
    # sns.kdeplot(filtered_data['起始车头时距'], color='green',fill=True, label='PDF')
    # sns.kdeplot(filtered_data['起始车头时距'],cumulative=True, label='CDF', bw_method=0.1)
    #
    # # sns.kdeplot(frequency_data, fill=True, label='PDF')
    # percentiles = [95, 99]
    # for percentile in percentiles:
    #     percentile_value = np.percentile(filtered_data['起始车头时距'].dropna(), percentile)
    #     cdf_percentile = percentileofscore(filtered_data['起始车头时距'].dropna(), percentile_value)
    #     line_color = 'red' if percentile == 95 else 'blue'  # Set different colors for each percentile
    #
    #     plt.axvline(x=percentile_value, color=line_color, linestyle='--',
    #                 label=f'{percentile}% Percentile (CDF: {cdf_percentile:.2f}%)')
    # plt.xlim(0)
    # plt.xlabel('Inter-platoon Time Headway(s)')
    # plt.ylabel('Density(Cumulative Probability)')
    # plt.legend(loc='upper left')
    # plt.show()

    # print(filtered_individual_data)
    # ###############################3创建一个字典来存储每个车道每个车队包含的车辆数
    vehicle_counts = {}

    # 遍历convoy_data
    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():

            lane = row['车道']
            convoy_data_list = row['车队']
            # 统计每个车队包含的车辆数
            if lane not in vehicle_counts:
                vehicle_counts[lane] = {}

            # 将每个车队的车辆数存储在字典中
            convoy_key = tuple(map(lambda x: tuple(x.items()), convoy_data_list))
            vehicle_counts[lane][convoy_key] = len(convoy_data_list)

    # # 打印或根据需要使用vehicle_counts
    # for lane, convoy_vehicle_counts in vehicle_counts.items():
    #     print(f"车道 {lane}:")
    #     for convoy, count in convoy_vehicle_counts.items():
    #         print(f"  车队 {convoy}: {count} 辆车")


    # 统计频率
    frequency_data =  []
    for lane, counts in vehicle_counts.items():
        for convoy, count in counts.items():
            frequency_data.append(count)

    #
    # ####统计车队间间距######
    intro_dataL1 =  []
    intro_dataL2 = []
    intro_dataL3 = []
    intro_dataL4 = []
    intro_dataL5 = []
    intro_dataL3r=[]
    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车道']=='L1':
                count = row['起始车头时距']
                intro_dataL1.append(count)
            if row['车道']=='L2':
                count = row['起始车头时距']
                intro_dataL2.append(count)
            if row['车道']=='L3':
                count = row['起始车头时距']
                intro_dataL3.append(count)
            if row['车道']=='L3r':
                count = row['起始车头时距']
                intro_dataL3r.append(count)
            if row['车道']=='L4':
                count = row['起始车头时距']
                intro_dataL4.append(count)
            if row['车道']=='L5':
                count = row['起始车头时距']
                intro_dataL5.append(count)
    intro_data=[list(intro_dataL1),list(intro_dataL2),list(intro_dataL3),list(intro_dataL3r),list(intro_dataL4),list(intro_dataL5)]
    # intro_data =  []
    # for convoy in filtered_data:
    #     # print(convoy)
    #     for index, row in convoy.iterrows():
    #         count = row['起始车头时距']
    #         intro_data.append(count)
    # intro_data = [item for item in intro_data if item is not None]

    #
    inter_data =  []
    inter_dataL1 =  []
    inter_dataL2 = []
    inter_dataL3 = []
    inter_dataL4 = []
    inter_dataL5 = []
    inter_dataL3r=[]
    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车道']=='L1':
                count = row['队内车头时距']
                inter_dataL1.extend(count)
            if row['车道']=='L2':
                count = row['队内车头时距']
                inter_dataL2.extend(count)
            if row['车道']=='L3':
                count = row['队内车头时距']
                inter_dataL3.extend(count)
            if row['车道']=='L3r':
                count = row['队内车头时距']
                inter_dataL3r.extend(count)
            if row['车道']=='L4':
                count = row['队内车头时距']
                inter_dataL4.extend(count)
            if row['车道']=='L5':
                count = row['队内车头时距']
                inter_dataL5.extend(count)
    inter_data=[list(inter_dataL1),list(inter_dataL2),list(inter_dataL3),list(inter_dataL3r),list(inter_dataL4),list(inter_dataL5)]


    # for convoy in filtered_data:
    #     # print(convoy)
    #     for index, row in convoy.iterrows():
    #         count = row['队内车头时距']
    #         inter_data.extend(count)
    # inter_data = [item for item in inter_data if item is not None]
    # inter_data = [item for item in inter_data if item is not None]


    platoonspeeddata = []
    platoonspeeddataL1 = []
    platoonspeeddataL2 = []
    platoonspeeddataL3 = []
    platoonspeeddataL4 = []
    platoonspeeddataL5 = []

    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车道'] == 'L1':
                count = row['队内速度']
                platoonspeeddataL1.extend(count)
            if row['车道'] == 'L2':
                count = row['队内速度']
                platoonspeeddataL2.extend(count)
            if row['车道'] == 'L3':
                count = row['队内速度']
                platoonspeeddataL3.extend(count)
            if row['车道'] == 'L4':
                count = row['队内速度']
                platoonspeeddataL4.extend(count)
            if row['车道'] == 'L5':
                count = row['队内速度']
                platoonspeeddataL5.extend(count)
            # platoonspeeddata.append(count)
    platoonspeeddata=[np.nanmean(platoonspeeddataL1),np.nanmean(platoonspeeddataL2),np.nanmean(platoonspeeddataL3),np.nanmean(platoonspeeddataL4),np.nanmean(platoonspeeddataL5)]

    # platoonspeeddata = [item for item in platoonspeeddata if item is not None]

    ind_data =  []
    for convoy in filtered_individual_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            count = row['车头时距']
            ind_data.append(count)
    ind_data = [item for item in ind_data if item is not None]


    ind_speeddata = []
    for convoy in filtered_individual_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            count = row['车速']
            ind_speeddata.append(count)
    ind_speeddata = [item for item in ind_speeddata if item is not None]

    platoonLCdata=0
    for lane, counts in vehicle_counts.items():
        for convoy, count in counts.items():
            for item in convoy:
                if item[2][1] != item[9][1]:
                    platoonLCdata += 1


    indLCdata = 0
    for convoy in filtered_individual_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车辆']['U_Lane'] != row['车辆']['D_Lane']:
                indLCdata += 1

    return  frequency_data,intro_data,inter_data,ind_data,platoonspeeddata,ind_speeddata,platoonLCdata,indLCdata

def information_extact_time2(data,timeperiod):
# data3=data3[(data3['VL_Ave']<7)]
    data.sort_values(by='D_Date_Time', inplace=True)
    data_select=data
    # 创建一个空列表来存储车队数据
    convoy_data = []
    individual_data=[]
    # 遍历不同的U_Lane值
    for lane in data_select['D_Lane'].unique():
        # 为当前车道筛选数据
        lane_data = data_select[(data_select['D_Lane']== lane)].sort_values(by='D_Date_Time')
        # 初始化用于跟踪车队数据的变量
        current_convoy = []
        current_convoy_start_headway = None
        current_intraplatoon=[]
        current_speed=[]
        # 遍历车道数据的行
        for index, row in lane_data.iterrows():
            # 检查U_Headway是否小于2.5
            if row['D_Headway'] < 4:
                # 如果current_convoy为空，更新起始车头时距
                # 添加前一行的数据到车队数据
                prev_row_index = lane_data.index.get_loc(index) - 1
                prev_row = lane_data.iloc[prev_row_index].to_dict() if prev_row_index >= 0 else None
                current_intraplatoon.append(row['D_Headway'])
                current_speed.append(row['D_Speed'])
                if prev_row:
                    if prev_row['D_Headway'] >4:
                            current_convoy.append(prev_row)
                            if current_convoy:
                                current_convoy_start_headway = prev_row['D_Headway']
                # 将行追加到current_convoy
                current_convoy.append(row.to_dict())
            else:
                # 检查current_convoy是否有数据
                if current_convoy:
                    # 将车队数据和起始车头时距追加到convoy_data
                    convoy_data.append({
                        '车道': lane,
                        '车队': current_convoy,
                        '起始车头时距': current_convoy_start_headway,
                        "队内车头时距": current_intraplatoon,
                        '队内速度':current_speed
                    })
                    # 重置current_convoy和起始车头时距
                    current_convoy = []
                    current_convoy_start_headway = None
                    current_intraplatoon = []
                    current_speed = []
                else:
                    individual_data.append({
                        '车道': lane,
                        '车辆': row,
                        "车头时距":row['D_Headway'],
                        '车速':row['D_Speed']
                    })
        # 检查是否在最后一行之后还有剩余的车队数据
        if current_convoy:
            convoy_data.append({
                '车道': lane,
                '车队': current_convoy,
                '起始车头时距': current_convoy_start_headway,
                "队内车头时距":current_intraplatoon,
                '队内速度': current_speed
            })
    # np.savetxt('zone4_platoon.csv', convoy_data, delimiter=",")
    # # 打印或根据需要使用convoy_data
    # for convoy in convoy_data:


    ################################5分钟分类##########################
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime, timedelta

    # 假设 convoy_data 是一个具有你提供的结构的字典列表
    # 如果不是，请根据实际的数据结构进行调整

    # 提取每个车队的第一辆车的 'U_Date_Time'

    first_vehicle_times = pd.to_datetime([convoy['车队'][0]['D_Date_Time'].split('.')[0] for convoy in convoy_data],
                                         errors='coerce')
    individual_vehicle_times = pd.to_datetime([vehicle['车辆']['D_Date_Time'].split('.')[0] for vehicle in individual_data],
                                         errors='coerce')

    # 设置时间间隔（例如，5分钟）
    interval = timedelta(minutes=10)

    # 定义起始时间和结束时间
    start_time = datetime.strptime('11:30:00', '%H:%M:%S')
    end_time = datetime.strptime('13:30:00', '%H:%M:%S')
    # 定义特定日期
    specific_date = datetime.strptime('2022-09-27', '%Y-%m-%d').date()
    start_time = datetime.combine(specific_date, start_time.time())
    end_time = datetime.combine(specific_date, end_time.time())
    # 创建一个空的字典，用于存储每个时间段的 DataFrame
    interval_data_dict = {}
    interval_individual_data_dict={}
    # 遍历每个时间段
    current_time = start_time
    while current_time < end_time:
        end_interval_time = current_time + interval


        # 从 convoy_data 中筛选在当前时间段内的所有车队数据
        selected_convoy_data = [convoy for convoy, time in zip(convoy_data, first_vehicle_times) if current_time <= time < end_interval_time]
        selected_individual_data = [vehicle for vehicle, time in zip(individual_data, individual_vehicle_times) if current_time <= time < end_interval_time]

        # 将选定的车队数据存储到 interval_data_dict 中
        if selected_convoy_data:
            interval_data_dict[current_time] = pd.DataFrame(selected_convoy_data)
        if selected_individual_data:
            interval_individual_data_dict[current_time] = pd.DataFrame(selected_individual_data)
        current_time = end_interval_time

    # 示例：访问 11:30-11:35 时间段的数据
    # if start_time in interval_data_dict:
    #     print("Data for 11:30-11:35:")
        # print(interval_data_dict[start_time])
    test_time = datetime.strptime(timeperiod, '%H:%M:%S')
    # test_time1 = datetime.strptime('11:40:00', '%H:%M:%S')
    # test_time2 = datetime.strptime('11:40:00', '%H:%M:%S')
    # test_time3 = datetime.strptime('11:40:00', '%H:%M:%S')
    # test_time4 = datetime.strptime('11:40:00', '%H:%M:%S')

    test_time = datetime.combine(specific_date, test_time.time())
    # test_time1 = datetime.strptime('12:18:00', '%H:%M:%S')
    # test_time1 = datetime.combine(specific_date, test_time1.time())
    # test_time2 = datetime.combine(specific_date, test_time2.time())
    # test_time3 = datetime.combine(specific_date, test_time3.time())
    # test_time4 = datetime.combine(specific_date, test_time4.time())

    # filtered_data = interval_data_dict[test_time]
    keys_to_get = [test_time]

    # 存储多个键对应的值
    filtered_data = []
    filtered_individual_data=[]
    # 逐个获取每个键对应的值
    for key in keys_to_get:
        if key in interval_data_dict:
            filtered_data.append(interval_data_dict[key])
            filtered_individual_data.append(interval_individual_data_dict[key])

    # print(filtered_individual_data)
    # ###############################3创建一个字典来存储每个车道每个车队包含的车辆数
    vehicle_counts = {}

    # 遍历convoy_data
    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():

            lane = row['车道']
            convoy_data_list = row['车队']
            # 统计每个车队包含的车辆数
            if lane not in vehicle_counts:
                vehicle_counts[lane] = {}

            # 将每个车队的车辆数存储在字典中
            convoy_key = tuple(map(lambda x: tuple(x.items()), convoy_data_list))
            vehicle_counts[lane][convoy_key] = len(convoy_data_list)

    # # 打印或根据需要使用vehicle_counts
    # for lane, convoy_vehicle_counts in vehicle_counts.items():
    #     print(f"车道 {lane}:")
    #     for convoy, count in convoy_vehicle_counts.items():
    #         print(f"  车队 {convoy}: {count} 辆车")


    # 统计频率
    frequency_data =  []
    frequency_dataL1 = []
    frequency_dataL2 = []
    frequency_dataL3 = []
    frequency_dataL3r = []
    frequency_dataL4 = []
    frequency_dataL5 = []
    for lane, counts in vehicle_counts.items():
        if lane=='L1':
            for convoy, count in counts.items():
                frequency_dataL1.append(count)
        if lane=='L2':
            for convoy, count in counts.items():
                frequency_dataL2.append(count)
        if lane=='L3':
            for convoy, count in counts.items():
                frequency_dataL3.append(count)
        if lane=='L3r':
            for convoy, count in counts.items():
                frequency_dataL3.append(count)
        if lane=='L4':
            for convoy, count in counts.items():
                frequency_dataL4.append(count)
        if lane=='L5':
            for convoy, count in counts.items():
                frequency_dataL5.append(count)
    frequency_data=[list(frequency_dataL1),list(frequency_dataL2),list(frequency_dataL3),list(frequency_dataL3r),list(frequency_dataL4),list(frequency_dataL5)]

    #
    # ####统计车队间间距######
    intro_dataL1 =  []
    intro_dataL2 = []
    intro_dataL3 = []
    intro_dataL3r = []
    intro_dataL4 = []
    intro_dataL5 = []
    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车道']=='L1':
                count = row['起始车头时距']
                intro_dataL1.append(count)
            if row['车道']=='L2':
                count = row['起始车头时距']
                intro_dataL2.append(count)
            if row['车道']=='L3':
                count = row['起始车头时距']
                intro_dataL3.append(count)
            if row['车道']=='L3r':
                count = row['起始车头时距']
                intro_dataL3r.append(count)
            if row['车道']=='L4':
                count = row['起始车头时距']
                intro_dataL4.append(count)
            if row['车道']=='L5':
                count = row['起始车头时距']
                intro_dataL5.append(count)
    intro_data=[list(intro_dataL1),list(intro_dataL2),list(intro_dataL3),list(intro_dataL3r),list(intro_dataL4),list(intro_dataL5)]

    # intro_data =  []
    # for convoy in filtered_data:
    #     # print(convoy)
    #     for index, row in convoy.iterrows():
    #         count = row['起始车头时距']
    #         intro_data.append(count)
    # intro_data = [item for item in intro_data if item is not None]

    #
    inter_data =  []
    inter_dataL1 =  []
    inter_dataL2 = []
    inter_dataL3 = []
    inter_dataL3r = []
    inter_dataL4 = []
    inter_dataL5 = []
    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车道']=='L1':
                count = row['队内车头时距']
                inter_dataL1.extend(count)
            if row['车道']=='L2':
                count = row['队内车头时距']
                inter_dataL2.extend(count)
            if row['车道']=='L3':
                count = row['队内车头时距']
                inter_dataL3.extend(count)
            if row['车道']=='L3r':
                count = row['队内车头时距']
                inter_dataL3r.extend(count)
            if row['车道']=='L4':
                count = row['队内车头时距']
                inter_dataL4.extend(count)
            if row['车道']=='L5':
                count = row['队内车头时距']
                inter_dataL5.extend(count)
    inter_data=[list(inter_dataL1),list(inter_dataL2),list(inter_dataL3),list(inter_dataL3r),list(inter_dataL4),list(inter_dataL5)]


    # for convoy in filtered_data:
    #     # print(convoy)
    #     for index, row in convoy.iterrows():
    #         count = row['队内车头时距']
    #         inter_data.extend(count)
    # inter_data = [item for item in inter_data if item is not None]


    platoonspeeddata = []
    platoonspeeddataL1 = []
    platoonspeeddataL2 = []
    platoonspeeddataL3 = []
    platoonspeeddataL4 = []
    platoonspeeddataL5 = []

    for convoy in filtered_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车道'] == 'L1':
                count = row['队内速度']
                platoonspeeddataL1.extend(count)
            if row['车道'] == 'L2':
                count = row['队内速度']
                platoonspeeddataL2.extend(count)
            if row['车道'] == 'L3':
                count = row['队内速度']
                platoonspeeddataL3.extend(count)
            if row['车道'] == 'L4':
                count = row['队内速度']
                platoonspeeddataL4.extend(count)
            if row['车道'] == 'L5':
                count = row['队内速度']
                platoonspeeddataL5.extend(count)
            # platoonspeeddata.append(count)
    platoonspeeddata=[np.nanmean(platoonspeeddataL1),np.nanmean(platoonspeeddataL2),np.nanmean(platoonspeeddataL3),np.nanmean(platoonspeeddataL4),np.nanmean(platoonspeeddataL5)]

    # platoonspeeddata = [item for item in platoonspeeddata if item is not None]

    ind_data =  []
    ind_dataL1 = []
    ind_dataL2 = []
    ind_dataL3 = []
    ind_dataL4 = []
    ind_dataL5 = []
    for convoy in filtered_individual_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车道'] == 'L1':
                count = row['车头时距']
                ind_dataL1.append(count)
            if row['车道'] == 'L2':
                count = row['车头时距']
                ind_dataL2.append(count)
            if row['车道'] == 'L3':
                count = row['车头时距']
                ind_dataL3.append(count)
            if row['车道'] == 'L4':
                count = row['车头时距']
                ind_dataL4.append(count)
            if row['车道'] == 'L5':
                count = row['车头时距']
                ind_dataL5.append(count)
    ind_data=[list(ind_dataL1),list(ind_dataL2),list(ind_dataL3),list(ind_dataL4),list(ind_dataL5)]
    # ind_data = [item for item in ind_data if item is not None]


    ind_speeddata = []
    for convoy in filtered_individual_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            count = row['车速']
            ind_speeddata.append(count)
    ind_speeddata = [item for item in ind_speeddata if item is not None]

    platoonLCdata=0

    for lane, counts in vehicle_counts.items():
        for convoy, count in counts.items():
            for item in convoy:
                if item[2][1]!=item[9][1]:
                    platoonLCdata+=1



    indLCdata=0
    for convoy in filtered_individual_data:
        # print(convoy)
        for index, row in convoy.iterrows():
            if row['车辆']['U_Lane']!=row['车辆']['D_Lane']:
                indLCdata+=1

    return  frequency_data,intro_data,inter_data,ind_data,platoonspeeddata,ind_speeddata,platoonLCdata,indLCdata

# frequency_data3,intro_data3,inter_data3,ind_data3,platoonspeeddata3,ind_speeddata3,LC3,LC13=information_extact_time1(data3)
# frequency_data4,intro_data4,inter_data4,ind_data4,platoonspeeddata4,ind_speeddata4,LC4,LC14=information_extact_time1(data4)
# frequency_data5,intro_data5,inter_data5,ind_data5,platoonspeeddata5,ind_speeddata5,LC5,LC15=information_extact_time1(data5)
# # frequency_data3s,intro_data3s,inter_data3s,ind_data3s,platoonspeeddata3s,ind_speeddata3s,LC3s,LC13s=information_extact_time2(data3)
# # frequency_data4s,intro_data4s,inter_data4s,ind_data4s,platoonspeeddata4s,ind_speeddata4s,LC4s,LC14s=information_extact_time2(data4)
# frequency_data5s,intro_data5s,inter_data5s,ind_data5s,platoonspeeddata5s,ind_speeddata5s,LC5s,LC15s=information_extact_time2(data5,'11:40:00')



import datetime

import datetime

import datetime


import datetime
from scipy.stats import expon,lognorm,norm
def remove_nonfinite_values(data):
    data = np.array(data, dtype=float)  # 强制转换数据类型为浮点数
    data = data[~np.isnan(data) & np.isfinite(data)]
    # data = data[(data >= 4) & (data <= 12)]
    return data

from scipy.stats import gaussian_kde
def extract_and_plot_histograms(data, start_time_str, end_time_str, interval_minutes=10):
    # 将字符串转换为时间对象
    start_time = datetime.datetime.strptime(start_time_str, '%H:%M:%S').time()
    end_time = datetime.datetime.strptime(end_time_str, '%H:%M:%S').time()

    # 初始化时间列表
    times = []
    current_time = start_time

    # 生成时间点
    while current_time <= end_time:
        times.append(current_time)
        current_time = (datetime.datetime.combine(datetime.date.today(), current_time) + datetime.timedelta(minutes=interval_minutes)).time()

    # 提取第四个列表数据并绘制直方图
    # plt.figure(figsize=(15, 6))
    for i, time in enumerate(times):
        # 将时间对象转换为字符串
        time_str = time.strftime('%H:%M:%S')
        frequency_data5s, intro_data5s, inter_data5s, ind_data5s, platoonspeeddata5s, ind_speeddata5s, LC5s, LC15s = information_extact_time2(data, time_str)

        # 绘制直方图
        # plt.hist(intro_data5s, bins=10, alpha=0.5, label=f'Time {time_str}')

        ones_array = [1] * len(ind_data5s)

        # 逐个将 ones_array 中的元素添加到 frequency_data5s 中
        for one in ones_array:
            frequency_data5s.append(np.ones(1))  #
        inter_data4=[item for sublist in inter_data5s for item in sublist]#######从这儿换数据！！！！！！！！
        # print(frequency_data5s)
        # # 使用核密度估计进行拟合
        # kde = gaussian_kde(inter_data4,bw_method=0.7)
        # x = np.linspace(min(inter_data4), max(inter_data4), 100)
        # plt.plot(x, kde(x), label=f'Time {time_str}')
        # plt.hist(inter_data4, bins=10, alpha=0.5, label=f'Time {time_str}')
        bin_edges = [0, 0.5, 1,1.5,2,2.5,3,3.5,4]
        # bin_edges = [4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12]
        #
        # bin_edges = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        time_obj = datetime.datetime.strptime(time_str, '%H:%M:%S')

        # 加上一小时
        new_time_obj = time_obj + timedelta(hours=1)

        # 提取新时间对象中的小时和分钟部分，并格式化为'%H:%M'的字符串
        new_time_str = new_time_obj.strftime('%H:%M:%S')

        # 使用格式化后的字符串作为标签
        sns.distplot(inter_data4, bins=bin_edges, kde=False, label=f'Time {time_str}-{new_time_str}')        # inter_data4_cleaned = remove_nonfinite_values(inter_data4)

        # inter_data4_cleaned = remove_nonfinite_values(inter_data4)
        # params = lognorm.fit(inter_data4_cleaned)
        # shape, loc, scale = params
        # xmin, xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = lognorm.pdf(x, shape, loc, scale)
        # plt.plot(x, p * 0.45 * len(inter_data4_cleaned), linewidth=2,
        #          label=f'Log-normal fit: shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f}')
        ##################inter platoon headway##################
        #
        inter_data4_cleaned = remove_nonfinite_values(inter_data4)
        loc, scale= norm.fit(inter_data4_cleaned)

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, loc, scale)
        plt.plot(x, p * 0.45 * len(inter_data4_cleaned), linewidth=2,
                 label=f'Normal fit: loc={loc:.2f}, scale={scale:.2f}')

        ##################intra platoon headway##################
        #
        # inter_data4_cleaned = remove_nonfinite_values(inter_data4)
        # loc, scale= expon.fit(inter_data4_cleaned)
        #
        # xmin, xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = expon.pdf(x, loc + 1, scale)
        # plt.xticks(range(1, 21))
        # plt.plot(x, p * 1.2*len(inter_data4_cleaned), linewidth=2,
        #          label=f'Exponential fit: loc={loc + 1:.2f}, scale={scale:.2f}')

        ##################platoon size##################

    plt.legend()
    plt.xlabel('Intra-Platoon Headway')
    plt.xlim((0,4))
    plt.ylim((0,120))

    plt.ylabel('Frequency')
    # plt.grid(True)
    output_path2 = f"{'/Users/pjl/Desktop/research_KUL/0704/platoonD2'}/{'figintraplatoonheadwayZ7.png'}"
    # 保存图表为高质量图片
    plt.savefig(output_path2, dpi=600, bbox_inches='tight')
    plt.show()

# 调用函数并传入参数
start_time_str = '11:30:00'
end_time_str = '13:20:00'

# data7L = data7[(data7['D_Date_Time'] >= start_time) & (data7['D_Date_Time'] <= end_time)]


extract_and_plot_histograms(data7, start_time_str, end_time_str, interval_minutes=60)


# print(inter_data3)
# print(inter_data3s)
# print(inter_data4)
# print(inter_data4s)
# print(inter_data5)
# print(inter_data5s)
#
# print(platoonspeeddata3)
# print(platoonspeeddata3s)
# print(platoonspeeddata4)
# print(platoonspeeddata4s)
# print(platoonspeeddata5)
# print(platoonspeeddata5s)
#
# print(LC3)
# print(LC3s)
# print(LC4)
# print(LC4s)
# print(LC5)
# print(LC5s)
#
# import matplotlib.pyplot as plt
# print(np.mean(platoonspeeddata3))
# print(np.std(platoonspeeddata3))
# #
# print(np.nanmean(inter_data3))
# print(np.nanstd(inter_data3))
#
# print(np.mean(ind_speeddata3))
# print(np.std(ind_speeddata3))
#
# print(np.mean(ind_speeddata3s))
# print(np.std(ind_speeddata3s))

# print(np.mean(frequency_data3))
# print(np.std(frequency_data3))
# print(np.percentile(frequency_data3, 95))
#
# print(np.nanmean(frequency_data3s))
# print(np.nanstd(frequency_data3s))
# print(np.nanpercentile(frequency_data3s, 95))
#
# print(np.mean(frequency_data4))
# print(np.std(frequency_data4))
# print(np.percentile(frequency_data4, 95))
#
# print(np.nanmean(frequency_data4s))
# print(np.nanstd(frequency_data4s))
# print(np.percentile(frequency_data4s, 95))
#
# print(np.mean(frequency_data5))
# print(np.std(frequency_data5))
# print(np.percentile(frequency_data5, 95))
#
# print(np.nanmean(frequency_data5s))
# print(np.nanstd(frequency_data5s))
# print(np.percentile(frequency_data5s, 95))

# print(np.nanmean(ind_speeddata5))
# print(np.nanstd(ind_speeddata5))
#
# print(np.mean(ind_speeddata5s))
# print(np.std(ind_speeddata5s))




# plt.show()
#
# # 创建一个四行三列的直方图子图
# fig, axs = plt.subplots(4, 3, figsize=(8, 8))
#
# # 绘制第一行直方图子图
# axs[0, 0].hist(frequency_data3, bins=10, color='skyblue', alpha=0.7, label='Z3(12:08-12:18)')
# axs[0, 0].hist(frequency_data3s, bins=10, color='salmon', alpha=0.4, label='Z3(12:38-12:48)')
# axs[0, 0].legend()
# axs[0, 0].set_xlim(0,40)
# axs[0, 0].set_xlabel("Platoon Size(Veh)")
#
# axs[0, 1].hist(frequency_data4, bins=10, color='skyblue', alpha=0.7, label='Z4(12:08-12:18)')
# axs[0, 1].hist(frequency_data4s, bins=10, color='salmon', alpha=0.4, label='Z4(12:38-12:48)')
# axs[0, 1].legend()
# axs[0, 1].set_xlim(0,40)
# axs[0, 1].set_xlabel("Platoon Size(Veh)")
#
# axs[0, 2].hist(frequency_data5, bins=10, color='skyblue', alpha=0.7, label='Z5(12:08-12:18)')
# axs[0, 2].hist(frequency_data5s, bins=10, color='salmon', alpha=0.4, label='Z5(12:38-12:48)')
# axs[0, 2].legend()
# axs[0, 2].set_xlim(0,40)
# axs[0, 2].set_xlabel("Platoon Size(Veh)")
#
# # 绘制第二行直方图子图
# axs[1, 0].hist(intro_data3, bins=10, color='skyblue', alpha=0.7, label='Z3(12:08-12:18)')
# axs[1, 0].hist(intro_data3s, bins=10, color='salmon', alpha=0.4, label='Z3(12:38-12:48)')
# axs[1, 0].legend()
# axs[1, 0].set_xlim(2,20)
# axs[1, 0].set_xlabel("Inter-platoon Headway(s)")
#
# axs[1, 1].hist(intro_data4, bins=10, color='skyblue', alpha=0.7, label='Z4(12:08-12:18)')
# axs[1, 1].hist(intro_data4s, bins=10, color='salmon', alpha=0.4, label='Z4(12:38-12:48)')
# axs[1, 1].legend()
# axs[1, 1].set_xlim(2,20)
# axs[1, 1].set_xlabel("Inter-platoon Headway(s)")
#
# axs[1, 2].hist(intro_data5, bins=10, color='skyblue', alpha=0.7, label='Z5(12:08-12:18)')
# axs[1, 2].hist(intro_data5s, bins=10, color='salmon', alpha=0.4, label='Z5(12:38-12:48)')
# axs[1, 2].legend()
# axs[1, 2].set_xlim(2,20)
# axs[1, 2].set_xlabel("Inter-platoon Headway(s)")
#
# # 绘制第三行直方图子图
# axs[2, 0].hist(inter_data3, bins=10, color='skyblue', alpha=0.7, label='Z3(12:08-12:18)')
# axs[2, 0].hist(inter_data3s, bins=10, color='salmon', alpha=0.4, label='Z3(12:38-12:48)')
# axs[2, 0].legend()
# axs[2, 0].set_xlim(0,4)
#
# axs[2, 0].set_xlabel("Intra-platoon Headway(s)")
#
# axs[2, 1].hist(inter_data4, bins=10, color='skyblue', alpha=0.7, label='Z4(12:08-12:18)')
# axs[2, 1].hist(inter_data4s, bins=10, color='salmon', alpha=0.4, label='Z4(12:38-12:48)')
# axs[2, 1].legend()
# axs[2, 1].set_xlim(0,4)
# axs[2, 1].set_xlabel("Intra-platoon Headway(s)")
#
# axs[2, 2].hist(inter_data5, bins=10, color='skyblue', alpha=0.7, label='Z5(12:08-12:18)')
# axs[2, 2].hist(inter_data5s, bins=10, color='salmon', alpha=0.4, label='Z5(12:38-12:48)')
# axs[2, 2].legend()
# axs[2, 2].set_xlim(0,4)
#
# axs[2, 2].set_xlabel("Intra-platoon Headway(s)")
#
# # 绘制第四行直方图子图
# axs[3, 0].hist(ind_data3, bins=5, color='skyblue', alpha=0.7, label='Z3(12:08-12:18)')
# axs[3, 0].hist(ind_data3s, bins=5, color='salmon', alpha=0.4, label='Z3(12:38-12:48)')
# axs[3, 0].legend()
# axs[3, 0].set_xlim(2,15)
# axs[3, 0].set_xlabel("Ind Headway(s)")
#
# axs[3, 1].hist(ind_data4, bins=5, color='skyblue', alpha=0.7, label='Z4(12:08-12:18)')
# axs[3, 1].hist(ind_data4s, bins=5, color='salmon', alpha=0.4, label='Z4(12:38-12:48)')
# axs[3, 1].legend()
# axs[3, 1].set_xlim(2,15)
# axs[3, 1].set_xlabel("Ind Headway(s)")
#
# axs[3, 2].hist(ind_data5, bins=5, color='skyblue', alpha=0.7, label='Z5(12:08-12:18)')
# axs[3, 2].hist(ind_data5s, bins=5, color='salmon', alpha=0.4, label='Z5(12:38-12:48)')
# axs[3, 2].legend()
# axs[3, 2].set_xlim(2,15)
# axs[3, 2].set_xlabel("Ind Headway(s)")
#
# # 去除所有子图的标题
# for ax in axs.flat:
#     ax.set_title('')
#
# plt.tight_layout()
# # output_path1 = f"{'/Users/pjl/Desktop/ITSC'}/{'fig3.png'}"
# # # 保存图表为高质量图片
# # plt.savefig(output_path1, dpi=600, bbox_inches='tight')
# plt.show()
