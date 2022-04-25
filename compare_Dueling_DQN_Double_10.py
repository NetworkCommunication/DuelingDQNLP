'''
基于Dueling DQN 比较凸优化优化飞行时间和Duelng DQN优化飞行时间
问题：修改环境中的周期数

比较2000轮次，每20个画一个。
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import math as mt
import matplotlib.pyplot as plt

env = UAV_MEC()
res = Res_plot_compare(env)


def plot_UAV_GT(UAV_trajectory_dueling_5, UAV_trajectory_dueling_9, Slot_dueling_5, Slot_dueling_9):
    for e in range(14990, 14991):

        x_dueling_5 = []
        y_dueling_5 = []
        z_dueling_5 = []

        x_dueling_9 = []
        y_dueling_9 = []
        z_dueling_9 = []

        sum_1 = 0
        sum_2 = 0

        for slot in range(env.N_slot):
            if slot < Slot_dueling_5[0, e]:
                x_dueling_5.append(UAV_trajectory_dueling_5[e, slot, 0])
                y_dueling_5.append(UAV_trajectory_dueling_5[e, slot, 1])
                z_dueling_5.append(UAV_trajectory_dueling_5[e, slot, 2])
            if slot < Slot_dueling_9[0, e]:
                x_dueling_9.append(UAV_trajectory_dueling_9[e, slot, 0])
                y_dueling_9.append(UAV_trajectory_dueling_9[e, slot, 1])
                z_dueling_9.append(UAV_trajectory_dueling_9[e, slot, 2])

        for i in range(Slot_dueling_5[0, e] - 1):
            sum_1 += np.sqrt(
                mt.pow(x_dueling_5[i + 1] - x_dueling_5[i], 2) + mt.pow(y_dueling_5[i + 1] - y_dueling_5[i],
                                                                        2) + mt.pow(z_dueling_5[i + 1] - z_dueling_5[i],
                                                                                    2))

        for i in range(Slot_dueling_9[0, e] - 1):
            sum_2 += np.sqrt(
                mt.pow(x_dueling_9[i + 1] - x_dueling_9[i], 2) + mt.pow(y_dueling_9[i + 1] - y_dueling_9[i],
                                                                        2) + mt.pow(z_dueling_9[i + 1] - z_dueling_9[i],
                                                                                    2))
        number = [sum_1, sum_2]
        index = ['5', '9']
        plt.bar(index, number, width=0.2)
        plt.title("The flying distance of the UAV is under two actions.")
        plt.ylabel("distance")
        plt.xlabel("action")
        # plt.legend()
        # plt.grid()
        plt.show()


if __name__ == '__main__':
    UAV_trajectory_dueling_5 = np.load("flyaction/5/UAV_trajectory_dueling.npy")
    Slot_dueling_5 = np.load("flyaction/5/Slot_dueling.npy")
    UAV_trajectory_dueling_9 = np.load("flyaction/9/UAV_trajectory_dueling.npy")
    Slot_dueling_9 = np.load("flyaction/9/Slot_dueling.npy")
    plot_UAV_GT(UAV_trajectory_dueling_5, UAV_trajectory_dueling_9, Slot_dueling_5, Slot_dueling_9)
