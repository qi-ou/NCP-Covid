# Evolution of those infected
# Author: Qi Ou
# Date: 25 Dec 2022
# Expanded from Zhihu
# https://zhuanlan.zhihu.com/p/115869172

import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

# 基于SEIR模型,知乎答主进行自主改进的“SEITR”模型介绍
# 所谓的新增人群“H”，指的是已被感染且正处于接受治疗时期的人群，
# 主要特征表现为已被感染，已过潜伏期，但不会进行传染，且正在被治疗。
# 我们也将I人群严格定义为被感染，已过潜伏期但未被医院收治无法接受治疗的人群。
# 同时引入了新的系数δ，表示I变为T的速率，主要受医院接诊速率及收治能力影响，也受发病后及时就医的时间影响。

# NCP 基于“SEITR”模型，改名“SEIHR”模型，H=住院
# 并且引入 重症率 severe 和 轻症转重症时间 Ts 这两个参数
# 这个模型假设所有重症都可以住院，并且都可以治愈，然而实际状况并非如此

# NCP 进一步创造"SHDR"模型，考虑医疗挤兑问题
# 针对每日新增的重症患者，每日医院治愈出院和死亡的人数空出的床位，计算每日能入院几名重症患者，其余算院外死亡
# 并显示每百万人病床数对最终死亡人数对影响


# N为人群总数，默认百万人
N =1000000
# β为传染率系数
beta = 0.5
# gamma_self为轻症自愈系数
gamma_self = 0.1
# gamma_hospital为重症治愈系数
gamma_hospital = 0.05
# severe 为重症率
severe = 0.1
# Te为潜伏转轻症天数
Te = 3
# Ts为轻症转重症天数
Ts = 5
# I_0为感染未住院的初始人数
I_0 = 1
# E_0为潜伏者的初始人数
E_0 = 0
# R_0为治愈者的初始人数
R_0 = 0
#T_0为治疗中的初始人数
T_0 = 0
# S_0为未感者的初始人数
S_0 = N - I_0 - E_0 - R_0 - T_0
# T为传播时间
T = 100
# INI为初始状态下的数组
INI = (S_0,E_0,I_0,R_0,T_0)
# bed 为每百万人可用床位
bed = 10000
# cure_rate = 住院总数中每天治愈出院的比例
cure_rate = gamma_hospital
# cure_rate = 住院总数中每天不治死亡的比例
death_rate = 0.


def calc_cumulative(new_time_series):
    cum = np.array([np.sum(new_time_series[:i + 1]) for i in np.arange(len(new_time_series))])
    return cum


def evolution(new_Severe, bed, cure_rate, death_rate):
    # 初始化 zero arrays
    in_hospital = np.zeros(len(new_Severe))  # 院内住院总人数
    recover = np.zeros(len(new_Severe))  # 每日医院治愈人数
    death_out_of_hospital = np.zeros(len(new_Severe))  # 每日新增由于医疗挤兑在院外死亡人数
    death_in_hospital = np.zeros(len(new_Severe))  # 每日新增院内不治死亡人数
    new_in_hospital = np.zeros(len(new_Severe))  # 每日新入院人数

    for i in np.arange(len(new_Severe)):
        # 新增入院
        if i == 0:
            in_hospital[i] = new_Severe[i]
        else:
            recover[i] = in_hospital[i-1] * cure_rate  # 部分出院，腾出床位
            death_in_hospital[i] = in_hospital[i-1] * death_rate  # 部分院内死亡， 腾出床位
            in_hospital[i] = in_hospital[i-1] - recover[i] - death_in_hospital[i] + new_Severe[i]  # 床位够多情况下的住院人数

        # 考虑医疗挤兑，床位不够
        if in_hospital[i] > bed:
            death_out_of_hospital[i] = in_hospital[i] - bed  # 床位不够，院外重症死亡
            new_in_hospital[i] = new_Severe[i] - death_out_of_hospital[i]  # 实际今日入院人数
            in_hospital[i] = bed  # 医院住满
        else:
            new_in_hospital[i] = new_Severe[i]  # 如果床位数够，就今日新增重症得以全部收治

    # 计算累计死亡和治愈
    cumu_death = calc_cumulative(death_in_hospital+death_out_of_hospital)
    cumu_recover = calc_cumulative(recover)
    final_death = cumu_death[-1]

    plt.plot(new_Severe, color='red', label='Newly_Severe')
    plt.plot(in_hospital, color='purple', label='In_Hospital')
    plt.plot(RES[:, 4], color='purple', label='In_Hospital_Ideal', linestyle=':')
    plt.plot(new_in_hospital, color='blue', label='Newly_Hospitalised')
    plt.plot(death_in_hospital, color='black', label='Daily_Death_In', linestyle='--')
    plt.plot(death_out_of_hospital, color='black', label='Daily_Death_Out', linestyle=':')
    plt.plot(cumu_death, color = 'black', label = 'Total_Death')
    plt.plot(recover, color='green', label='Daily_Recover', linestyle='--')
    plt.plot(cumu_recover, color='green', label='Total_Recover')

    plt.title('“SHDR” Model, bed={}, death={}'.format(bed, int(final_death)))
    plt.legend(loc='right')
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.xlim((0, T))
    # plt.ylim((10**1,10**5))
    # plt.yscale('log')
    plt.show()

    return final_death


def funcSEIHR(inivalue, _):
    # 微分公式
    Y = np.zeros(5)
    X = inivalue
    # 易感变化
    Y[0] = - (beta * X[0] * (X[2]+X[1])) / N
    # 潜伏变化
    Y[1] = (beta * X[0] * (X[2]+X[1])) / N - X[1] / Te
    # 轻症变化
    Y[2] = X[1] / Te - X[2] * severe / Ts - gamma_self * X[2] * (1-severe)
    # 治愈变化
    Y[3] = gamma_hospital * X[4] + gamma_self * X[2] * (1-severe)
    # 重症变化
    Y[4] = X[2] * severe / Ts - gamma_hospital * X[4]
    return Y


def integrate_SEIHR(T, INI):
    # 运行SEIHR模型
    T_range = np.arange(0, T + 1)
    RES = spi.odeint(funcSEIHR, INI, T_range)

    plt.plot(RES[:, 0], color='darkblue', label='Susceptible')
    plt.plot(RES[:, 1], color='orange', label='Exposed')
    plt.plot(RES[:, 2], color='red', label='Infection')
    plt.plot(RES[:, 3], color='green', label='Recovery')
    plt.plot(RES[:, 4], color='purple', label='Hospitalised')
    plt.hlines(bed, 0, 100, color='black', label='Bed')

    plt.title('“SEIHR” Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.xlim((0, T))
    plt.show()

    # 以下通过未感染人数的递减，计算每日新感染人数曲线，并通过转轻症时间、重症率和转重症时间，计算每日新增重症曲线
    S = RES[:,0]
    new_E = S[:-1] - S[1:]
    new_I = np.concatenate((np.zeros(Te), new_E))
    new_Severe = np.concatenate((np.zeros(Ts), new_I)) * severe

    plt.plot(RES[:, 0], color='darkblue', label='Susceptible')
    plt.plot(RES[:, 1], color='orange', label='Exposed')
    plt.plot(RES[:, 2], color='red', label='Infection')
    plt.plot(RES[:, 3], color='green', label='Recovery')
    plt.plot(RES[:, 4], color='purple', label='Hospitalised')
    plt.plot(new_E, color='orange',label='Newly_Exposed', linestyle='--')
    plt.plot(new_I, color='red', label='Newly_Infected', linestyle='--')
    plt.plot(new_Severe, color='red', label='Newly_Severe', linestyle=':')
    plt.hlines(bed, 0, 100, color='black', label='Bed')

    plt.title('“SEIHR” Model')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.xlim((0, T))
    plt.show()

    return RES, new_Severe


if __name__ == '__main__':
    # 通过传播模型得出每日新增重症曲线
    RES, new_Severe = integrate_SEIHR(T, INI)

    # 基于每日新增重症曲线，探究床位对于死亡人数对影响
    death_list = []
    bed_list = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    for bed in bed_list:
        death = evolution(new_Severe, bed, cure_rate, death_rate)
        death_list.append(death)

    # 画图床位对于死亡人数对影响
    plt.plot(bed_list, death_list)
    plt.title("Death against Beds")
    plt.xlabel("Bed / Million Population")
    plt.ylabel("Death")
    plt.show()







