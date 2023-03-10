#https://zhuanlan.zhihu.com/p/115869172
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

# N为人群总数
N = 10000
# β为传染率系数
beta = 0.25
# gamma为恢复率系数
gamma = 0.05
# I_0为感染者的初始人数
I_0 = 1
# R_0为治愈者的初始人数
R_0 = 0
# S_0为易感者的初始人数
S_0 = N - I_0 - R_0
# T为传播时间
T = 150

# INI为初始状态下的数组
INI = (S_0,I_0,R_0)


def funcSIR(inivalue,_):
 Y = np.zeros(3)
 X = inivalue
 # 易感个体变化
 Y[0] = - (beta * X[0] * X[1]) / N
 # 感染个体变化
 Y[1] = (beta * X[0] * X[1]) / N - gamma * X[1]
 # 治愈个体变化
 Y[2] = gamma * X[1]
 return Y

T_range = np.arange(0,T + 1)

RES = spi.odeint(funcSIR,INI,T_range)


plt.plot(RES[:,0],color = 'darkblue',label = 'Susceptible',marker = '.')
plt.plot(RES[:,1],color = 'red',label = 'Infection',marker = '.')
plt.plot(RES[:,2],color = 'green',label = 'Recovery',marker = '.')
plt.title('SIR Model')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()