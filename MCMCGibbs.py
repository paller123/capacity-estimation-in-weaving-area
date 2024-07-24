# import pymc3 as pm
# import numpy as np
# import arviz as az
# import matplotlib.pyplot as plt
# import pymc3 as pm
# import numpy as np
# import matplotlib.pyplot as plt
#
# # # 生成模拟数据
# # np.random.seed(42)
# # data1 = np.random.normal(0, 1, 100)
# # data2 = np.random.normal(0, 1, 100)
# #
# # # 定义 Gibbs 采样模型
# # with pm.Model() as model:
# #     # 定义先验分布
# #     theta1 = pm.Normal('theta1', mu=0, sd=1)
# #     theta2 = pm.Normal('theta2', mu=0, sd=1)
# #
# #     # 定义似然函数
# #     likelihood1 = pm.Normal('likelihood1', mu=theta1, sd=1, observed=data1)
# #     likelihood2 = pm.Normal('likelihood2', mu=theta2, sd=1, observed=data2)
# #
# #     # 定义 Gibbs 采样步骤
# #     step1 = pm.Metropolis(vars=[theta1])  # 从 theta1 的条件分布中采样
# #     step2 = pm.Metropolis(vars=[theta2])  # 从 theta2 的条件分布中采样
# #
# #     # 运行 Gibbs 采样
# #     trace = pm.sample(1000, tune=1000, step=[step1, step2], cores=1, progressbar=False)
# #
# # # 绘制后验分布
# # pm.traceplot(trace)
# # plt.show()
# import pymc3 as pm
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Simulated data
# np.random.seed(42)
# x = np.linspace(0, 10, 50)
# true_slope = 2
# true_intercept = 1
# y = true_slope * x + true_intercept + np.random.normal(0, 1, size=len(x))
# print(x)
# # Define the Bayesian model
# with pm.Model() as model:
#     # Priors
#     slope = pm.Normal('slope', mu=0, sd=10)
#     intercept = pm.Normal('intercept', mu=0, sd=10)
#
#     # Likelihood
#     likelihood = pm.Normal('y', mu=slope * x + intercept, sd=1, observed=y)
#
#     # MCMC sampling
#     trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)
#
# # Extract samples from the trace
# posterior_samples = pm.sample_posterior_predictive(trace, samples=1000, model=model)
#
# # Plot the scatter plot with the posterior predictive samples
# plt.scatter(x, y, label='Observed data')
#
# # Plot posterior predictive samples
# for i in range(1000):
#     plt.plot(x, posterior_samples['y'][i], color='gray', alpha=0.01)
#
# # Plot different percentiles with their own confidence intervals
# percentiles = [10, 50, 90]
# colors = ['red', 'green', 'blue']
# for p, color in zip(percentiles, colors):
#     interval = pm.hdi(posterior_samples['y'][:, :, None], hdi_prob=(1 - p / 100))
#     plt.fill_between(x, interval[:, 0], interval[:, 1], color=color, alpha=0.3, label=f'{p}th Percentile CI')
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Scatter Plot with Posterior Predictive Samples and Percentiles')
# plt.legend()
# plt.show()
#
#
#
#
#
#
#
#
import matplotlib.pyplot as plt

# 创建数据
x_data = [1, 2, 3, 4]
y1_data = [1, 2, 1, 3]
y2_data = [2, 1, 3, 4]

# 创建图表和轴
fig, ax1 = plt.subplots()

# 绘制第一个数据集在上方
ax1.plot(x_data, y1_data, 'b-', label='Plot 1 (Top)')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis for Plot 1 (Top)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim(-2, 2)  # 调整y轴范围
ax1.set_yticks([0, 1, 2])  # 设置y轴刻度为正整数
ax1.legend(loc='upper left')

# 创建第二个y轴，并绘制第二个数据集在下方
ax2 = ax1.twinx()
ax2.plot(x_data, y2_data, 'r-', label='Plot 2 (Bottom)')
ax2.set_ylabel('Y-axis for Plot 2 (Bottom)', color='r')
ax2.tick_params('y', colors='r')
ax2.set_ylim(0, 4)
ax2.set_yticks([0, 1, 2, 3, 4])  # 设置y轴刻度为正整数
ax2.legend(loc='lower right')

# 显示图表
plt.show()



