# Evolution of those infected
# Author: Qi Ou
# Date: 27 Dec 2022
# Developed based on Zhihu article: https://zhuanlan.zhihu.com/p/115869172

import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# NCP 独创 SIR_SHDC 模型
# 在传统疫情传播模型的基础上，嵌入病程模型和医院模型
# 通过 传播/重症/挤兑 三个阶段
# 推算 重症_Severe / 住院_Hospialised / 死亡_Dead / 治愈_Cured 曲线
# 旨在通过预测医疗资源的缺口，指导资源对接，以减少死亡人数

# 1. 传播
# 基于传统SIR模型，计算每日新增感染曲线
# 忽略常用的SEIR模型中的Exposed人群，因为此次omicron几乎没有潜伏期
# 在传播模型里，不区分轻重症，不涉及医院情况
# 只将人群分为 易感S(Susceptible), 感染I(Infectious)，不再具有传染性也不再易感U(Unsusceptible)（包括康复免疫+死亡）
# 通过 现存易感人群S、现存感染人群I，和传播率beta，计算 每日新增感染曲线newly_infected (即易感S的微分曲线）

# 2. 重症
# 每日新增感染曲线newly_infected，
# 将重症人群的分为 有基础病重症(severe_sick) / 没有基础病重症(severe_not_sick)三个群体
# 两个群体进入重症的时间高峰不同
# 两条重症曲线相加，汇入 重症_Severe 人群

# 2. 挤兑
# 重症_Severe 人群排队住院
# 根据床位，每日出院比例，每日院内死亡比例，计算出院外重症人数
# 院外重症几日不治后，成为院外死亡
# 院内/院外死亡曲线相加，成为总死亡曲线


##############
# 疫情传播参数
##############
# N为人群总数
N =22000
# I_0为感染未住院的初始人数
I_0 = 1/1000
# β = 传染率系数
beta = 0.55
# Tr(recover) = 轻症恢复所需天数
Tr = 10
# Ta(again) = 恢复人群转易感天数
Ta = 10
# again = 恢复人群转易感比例
again = 0.02
# d = 12月7号在传播过程的第几天
d = 28


##############
# 重症参数
##############
# severe_sick = = 总人群中，有基础疾病，一感染即重症的比例， 他们的Ts=0
severe_sick = 0.001
# severe_not_sick = 总人群中，没有基础疾病，但会重症的人的比例
severe_not_sick = 0.003
# Ts_sick = 对于有基础疾病的人群（severe_sick）来说， 轻症转重症的平均天数
Ts_sick = 3
# Ts_not_sick = 对于没有基础疾病的人群（severe_not_sick）来说， 轻症转重症的平均天数
Ts_not_sick = 7


##############
# 医院参数
##############
# bed 为可用床位
bed = 1000
# cure_rate = 住院总数中每天治愈出院的比例
cure_rate = 0.07
# cure_rate = 住院总数中每天不治死亡的比例
death_rate = 0.01
# Td_out = 院外重症人群的存活天数（不管有/没有基础疾病， 对于severe_sick + severe_not_sick 人群）
Td_out = 3


########################
# 其他默认疫情传播初始化参数
########################
# R_0为治愈者的初始人数
R_0 = 0
# S_0为易感者的初始人数
S_0 = N - I_0 - R_0
# T为传播时间
T = 365
# T_range = 时间序列的天数
T_range = np.arange(0, T + 1)
# INI为初始状态下的数组
INI = (S_0,I_0,R_0)


def calc_cumulative(new_time_series):
    cum = np.array([np.sum(new_time_series[:i + 1]) for i in np.arange(len(new_time_series))])
    return cum


def hospital(new_Severe, bed):
    # 初始化 zero arrays
    severe_queue = np.zeros(len(new_Severe))  # 院外待住院的重症
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
            in_hospital[i] = in_hospital[i-1] - recover[i] - death_in_hospital[i] + new_Severe[i] + severe_queue[i-1]  # 床位够多情况下的住院人数

        # 考虑医疗挤兑，床位不够
        if in_hospital[i] > bed:
            severe_queue[i] = in_hospital[i] - bed  # 床位不够，院外重症排队
            in_hospital[i] = bed  # 医院住满
            new_in_hospital[i] = new_Severe[i] + severe_queue[i-1] - severe_queue[i]  # 实际今日入院人数
            death_out_of_hospital[i] = severe_queue[i] / Td_out   # 院外重症死亡
            severe_queue[i] = severe_queue[i] - death_out_of_hospital[i]
        else:
            new_in_hospital[i] = new_Severe[i] + severe_queue[i-1] # 如果床位数够，就今日新增重症得以全部收治

    # 计算累计死亡和治愈
    cumu_death = calc_cumulative(death_in_hospital+death_out_of_hospital)
    cumu_recover = calc_cumulative(recover)
    final_death = cumu_death[-1]
    cumu_severe = calc_cumulative(new_Severe)

    plt.plot(cumu_severe, color='red', label='Total_Severe')
    plt.plot(cumu_recover, color='green', label='Total_Cured')
    plt.plot(cumu_death, color='black', label='Total_Death')
    plt.plot(in_hospital, color='purple', label='In_Hospital')

    # plt.plot(new_in_hospital, color='blue', label='Newly_Hospitalised', linestyle=':')
    plt.plot(new_Severe, color='red', label='Newly_Severe', linestyle=':')
    plt.plot(death_in_hospital, color='black', label='Daily_Death_In', linestyle='--')
    plt.plot(death_out_of_hospital, color='black', label='Daily_Death_Out', linestyle=':')
    plt.plot(recover, color='green', label='Daily_Cured', linestyle='--')

    plt.title('“SHDC” Model, population={}m, bed={}k, death={}'.format(int(N/1000000), int(bed/1000), int(final_death)))
    plt.legend(loc='right')
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.xlim((0, T))
    # plt.ylim((10**1,10**4))
    # plt.yscale('log')
    plt.show()

    return final_death


def funcSIR(inivalue,_):
    Y = np.zeros(3)
    X = inivalue
    # 易感个体变化
    Y[0] = - (beta * X[0] * X[1]) / N + X[2] / Ta * again
    # 感染个体变化
    Y[1] = (beta * X[0] * X[1]) / N - X[1] / Tr
    # 治愈个体变化
    Y[2] = X[1] / Tr - X[2] / Ta * again
    return Y


def integrate_SIU(T_range, INI):
    # 运行SIU（=SIU）模型
    RES = spi.odeint(funcSIR, INI, T_range)

    # 以下通过未感染人数的递减，计算每日新感染人数曲线
    S = RES[:,0]
    new_I = S[:-1] - S[1:]


    plt.plot(RES[:, 0], color='darkblue', label='Susceptible')
    plt.plot(RES[:, 1], color='red', label='Infection')
    plt.plot(RES[:, 2], color='green', label='Unsusceptible')
    plt.plot(new_I, color='darkblue', label='Newly_Infected', linestyle="--")
    plt.scatter(8+d, 2000, color='darkblue', label='Beijing_peak_Dec15_new2million')
    plt.scatter(8+d, N * 0.6 - RES[:, 2][8+d], color='red', marker='s', s=50, label='Beijing_peak_Dec15_cumu_60%-Recoverd')
    plt.scatter(d, 20, color='red', marker='s', s=50, label='Beijing_Dec7_cumu_20k')
    plt.scatter(d, 10, color='darkblue', label='Beijing_Dec7_new10k')

    plt.title('SIR Model, beta={}, Tr={}, Ta={}, again={}%'.format(beta, Tr, Ta, again*100))
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Number / k')
    plt.xlim((0, T))
    plt.show()

    return RES, new_I


def infection_to_severe(new_I):
    # 对 每日新感染人数曲线 和 轻症转重症的时间正态分布曲线 进行卷积，得到新增重症曲线
    newly_severe_sick = np.convolve(new_I * severe_sick, poisson.pmf(T_range, Ts_sick))
    newly_severe_not_sick = np.convolve(new_I * severe_not_sick, poisson.pmf(T_range, Ts_not_sick))
    newly_severe = newly_severe_sick + newly_severe_not_sick

    # 画图
    fig, ax = plt.subplots(2,sharex='all')
    ax[0].plot(new_I, color='darkblue', label='Newly_Infected', linestyle="--")
    ax[1].plot(newly_severe_sick, label="Newly_Severe_Sick ({:.1%}), Ts={}".format(severe_sick, Ts_sick))
    ax[1].plot(newly_severe_not_sick, color='green', label="Newly_Severe_not_Sick ({:.1%}), Ts={}".format(severe_not_sick, Ts_not_sick))
    ax[1].plot(newly_severe, color='red', label="Newly_Severe")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel("Number")
    ax[1].set_ylabel("Number")
    ax[1].set_xlabel("Days")
    plt.xlim((0, T))
    plt.suptitle("Infection to Severe")
    plt.show()
    return newly_severe


if __name__ == '__main__':
    # 通过传播模型得出每日新增感染曲线
    RES, new_I = integrate_SIU(T_range, INI)
    # newly_severe = infection_to_severe(new_I)
    # final_death = hospital(newly_severe, bed)

    # # 基于每日新增重症曲线，探究床位对于死亡人数对影响
    # death_list = []
    # bed_list = [2000, 4000, 6000]
    # for bed in bed_list:
    #     death = hospital(newly_severe, bed)
    #     death_list.append(death)
    #
    # # 画图床位对于死亡人数对影响
    # plt.plot(bed_list, death_list)
    # plt.title("Death against Beds")
    # plt.xlabel("Bed / Million Population")
    # plt.ylabel("Death")
    # plt.show()
    #






