# https://zhuanlan.zhihu.com/p/115869172
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

# N为人群总数
N =11000000
# β为传染率系数
beta = 0.01
# gamma为恢复率系数
gamma = 0.1
#δ为受到治疗系数
δ = 0.3
# Te为疾病潜伏期
Te = 5
# I_0为感染未住院的初始人数
I_0 = 4058
# E_0为潜伏者的初始人数
E_0 = 3178
# R_0为治愈者的初始人数
R_0 = 91750
#T_0为治疗中的初始人数
T_0 = 53941
# S_0为易感者的初始人数
S_0 = N - I_0 - E_0 - R_0 - T_0
# T为传播时间
T = 100

# INI为初始状态下的数组
INI = (S_0,E_0,I_0,R_0,T_0)

def funcSEIR(inivalue,_):
 Y = np.zeros(5)
 X = inivalue
 # 易感个体变化
 Y[0] = - (beta * X[0] *( X[2]+X[1])) / N
 # 潜伏个体变化
 Y[1] = (beta * X[0] *( X[2]+X[1])) / N - X[1] / Te
 # 感染未住院
 Y[2] = X[1] / Te - δ * X[2]
 # 治愈个体变化
 Y[3] = gamma * X[4]
 #治疗中个体变化
 Y[4] = δ* X[2] - gamma* X[4]
 return Y

T_range = np.arange(0,T + 1)

RES = spi.odeint(funcSEIR,INI,T_range)

plt.plot(RES[:,0],color = 'darkblue',label = 'Susceptible',marker = '.')
plt.plot(RES[:,1],color = 'orange',label = 'Exposed',marker = '.')
plt.plot(RES[:,2],color = 'red',label = 'Infection',marker = '.')
plt.plot(RES[:,3],color = 'green',label = 'Recovery',marker = '.')
plt.plot(RES[:,4],color = 'purple',label = 'Under Treatment',marker = '.')

plt.title('“SEITR” Model')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()