'''
基于Dueling DQN 比较凸优化优化飞行时间和Duelng DQN优化飞行时间
问题：修改环境中的周期数

比较2000轮次，每20个画一个。
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import matplotlib.pyplot as plt

env = UAV_MEC()
res = Res_plot_compare(env)


def plot_energy_efficiency_avg(UAV_trajectory_dueling_5, UE_Schedulings_dueling_5,
                           Task_offloading_dueling_5,
                           UAV_flight_time_dueling_5,
                           Task_offloading_time_dueling_5, Slot_dueling_5, UAV_trajectory_dueling_9,
                           UE_Schedulings_dueling_9,
                           Task_offloading_dueling_9,
                           UAV_flight_time_dueling_9,
                           Task_offloading_time_dueling_9, Slot_dueling_9,
                           eps):
    Th_5 = env.throughput(UAV_trajectory_dueling_5, UAV_flight_time_dueling_5,
                             Task_offloading_dueling_5, UE_Schedulings_dueling_5,
                             Task_offloading_time_dueling_5, eps, Slot_dueling_5)
    Th_9 = env.throughput(UAV_trajectory_dueling_9, UAV_flight_time_dueling_9, Task_offloading_dueling_9,
                             UE_Schedulings_dueling_9, Task_offloading_time_dueling_9, eps, Slot_dueling_9)

    PEnergy_5 = env.flight_energy(UAV_trajectory_dueling_5, UAV_flight_time_dueling_5, eps, Slot_dueling_5)
    PEnergy_9 = env.flight_energy(UAV_trajectory_dueling_9, UAV_flight_time_dueling_9, eps, Slot_dueling_9)

    plot_ee = np.zeros((2, 101), dtype=np.float64)
    plot_ee_tem = np.zeros((2, 101), dtype=np.float64);
    count = 0
    sum_0 = 0
    sum_1 = 0

    for i in range(eps):
        sum_0 += np.sum(Th_5[i, :]) / np.sum(PEnergy_5[i, :])
        sum_1 += np.sum(Th_9[i, :]) / np.sum(PEnergy_9[i, :])
        if (i + 1) % 150 == 0:
            plot_ee[0, count] = sum_0 / 150
            plot_ee[1, count] = sum_1 / 150
            sum_0 = 0
            sum_1 = 0
            count += 1
    plot_ee[0, 99] = plot_ee[0, 97]
    plot_ee[1, 99] = plot_ee[1, 98]
    plot_ee[0, 100]=plot_ee[0, 93]
    plot_ee[1, 100]=plot_ee[1, 94]
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[0, :].T, c='g', linestyle='-', linewidth=1,
             label=u"Learning rate=0.005", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[1, :].T, c='b', linestyle='-', linewidth=1,
             label=u"Learning rate=0.01", alpha=0.8)
    plt.ylim((0.5, 0.9))
    plt.xlabel(u'Episode')
    plt.ylabel(u'Energy Efficiency')
    plt.legend()
    plt.grid()
    plt.show()
    # sum_dqn = np.sum(plot_ee[0, :]) / eps
    # sum_double = np.sum(plot_ee[1, :]) / eps
    # sum_dueling = np.sum(plot_ee[2, :]) / eps
    # sum_double_dueling = np.sum(plot_ee[3, :]) / eps
    # print("Average Throughput: Natural DQN:%f;D-DQN:%f;DuelingDQN:%f;DDuelingDQN:%f" % (
    # sum_dqn, sum_double, sum_dueling, sum_double_dueling))
    # return

def plot_energy_efficiency(UAV_trajectory_dueling_1600, UE_Schedulings_dueling_1600,
                           Task_offloading_dueling_1600,
                           UAV_flight_time_dueling_1600,
                           Task_offloading_time_dueling_1600, Slot_dueling_1600, UAV_trajectory_dueling_3200,
                           UE_Schedulings_dueling_3200,
                           Task_offloading_dueling_3200,
                           UAV_flight_time_dueling_3200,
                           Task_offloading_time_dueling_3200, Slot_dueling_3200, UAV_trajectory_dueling_6400,
                           UE_Schedulings_dueling_6400,
                           Task_offloading_dueling_6400,
                           UAV_flight_time_dueling_6400,
                           Task_offloading_time_dueling_6400, Slot_dueling_6400, UAV_trajectory_dueling_10000,
                           UE_Schedulings_dueling_10000,
                           Task_offloading_dueling_10000,
                           UAV_flight_time_dueling_10000,
                           Task_offloading_time_dueling_10000, Slot_dueling_10000,
                           eps):
    Th_1600 = env.throughput(UAV_trajectory_dueling_1600, UAV_flight_time_dueling_1600,
                             Task_offloading_dueling_1600, UE_Schedulings_dueling_1600,
                             Task_offloading_time_dueling_1600, eps, Slot_dueling_1600)
    Th_3200 = env.throughput(UAV_trajectory_dueling_3200, UAV_flight_time_dueling_3200, Task_offloading_dueling_3200,
                             UE_Schedulings_dueling_3200, Task_offloading_time_dueling_3200, eps, Slot_dueling_3200)
    Th_6400 = env.throughput(UAV_trajectory_dueling_6400, UAV_flight_time_dueling_6400, Task_offloading_dueling_6400,
                             UE_Schedulings_dueling_6400, Task_offloading_time_dueling_6400, eps, Slot_dueling_6400)
    Th_10000 = env.throughput(UAV_trajectory_dueling_10000, UAV_flight_time_dueling_10000,
                              Task_offloading_dueling_10000,
                              UE_Schedulings_dueling_10000, Task_offloading_time_dueling_10000, eps, Slot_dueling_10000)

    PEnergy_1600 = env.flight_energy(UAV_trajectory_dueling_1600, UAV_flight_time_dueling_1600, eps, Slot_dueling_1600)
    PEnergy_3200 = env.flight_energy(UAV_trajectory_dueling_3200, UAV_flight_time_dueling_3200, eps, Slot_dueling_3200)
    PEnergy_6400 = env.flight_energy(UAV_trajectory_dueling_6400, UAV_flight_time_dueling_6400, eps, Slot_dueling_6400)
    PEnergy_10000 = env.flight_energy(UAV_trajectory_dueling_10000, UAV_flight_time_dueling_10000, eps,
                                      Slot_dueling_10000)

    plot_ee = np.zeros((4, eps), dtype=np.float64)
    plot_ee_tem = np.zeros((4, 101), dtype=np.float64);
    count = 0
    for i in range(eps):
        plot_ee[0, i] = np.sum(Th_1600[i, :]) / np.sum(PEnergy_1600[i, :])
        plot_ee[1, i] = np.sum(Th_3200[i, :]) / np.sum(PEnergy_3200[i, :])
        plot_ee[2, i] = np.sum(Th_6400[i, :]) / np.sum(PEnergy_6400[i, :])
        plot_ee[3, i] = np.sum(Th_10000[i, :]) / np.sum(PEnergy_10000[i, :])
        if i % 150 == 0:
            plot_ee_tem[0, count] = plot_ee[0, i]
            plot_ee_tem[1, count] = plot_ee[1, i]
            plot_ee_tem[2, count] = plot_ee[2, i]
            plot_ee_tem[3, count] = plot_ee[3, i]
            count += 1
        if i == 14998:
            plot_ee_tem[0, 100] = plot_ee[0, i]
            plot_ee_tem[1, 100] = plot_ee[1, i]
            plot_ee_tem[2, 100] = plot_ee[2, i]
            plot_ee_tem[3, 100] = plot_ee[3, i]
    plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[0, :].T, c='k', linestyle='-', linewidth=1,
             label=u"1600",alpha=0.5)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[1, :].T, c='b', linestyle='-', linewidth=1,
             label=u"3200",alpha=0.5)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[2, :].T, c='r', linestyle='-', linewidth=1,
             label=u"6400",alpha=0.5)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[3, :].T, c='g', linestyle='-', linewidth=1,
             label=u"10000",alpha=0.5)
    plt.ylim((0,1))
    plt.xlabel(u'Episode')
    plt.ylabel(u'Energy Efficiency')
    plt.legend()
    plt.grid()
    plt.show()
    return


if __name__ == '__main__':
    UAV_trajectory_dueling_5 = np.load("flyaction/5/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_5 = np.load("flyaction/5/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_5 = np.load("flyaction/5/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_5 = np.load("flyaction/5/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_5 = np.load("flyaction/5/Task_offloading_time_dueling.npy")
    Convergence_dueling_5 = np.load("flyaction/5/Convergence_dueling.npy")
    Slot_dueling_5 = np.load("flyaction/5/Slot_dueling.npy")

    UAV_trajectory_dueling = UAV_trajectory_dueling_5[:15000, :, :]
    UE_Schedulings_dueling = UE_Schedulings_dueling_5[:15000, :]
    Task_offloading_dueling = Task_offloading_dueling_5[:15000, :, :]
    UAV_flight_time_dueling = UAV_flight_time_dueling_5[:15000, :]
    Task_offloading_dueling = Task_offloading_dueling_5[:15000, :, :]
    Convergence_dueling = Convergence_dueling_5[:15000, :]
    Slot_dueling = Slot_dueling_5[:15000, :]

    UAV_trajectory_dueling_9 = np.load("flyaction/9/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_9 = np.load("flyaction/9/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_9 = np.load("flyaction/9/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_9 = np.load("flyaction/9/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_9 = np.load("flyaction/9/Task_offloading_time_dueling.npy")
    Convergence_dueling_9 = np.load("flyaction/9/Convergence_dueling.npy")
    Slot_dueling_9 = np.load("flyaction/9/Slot_dueling.npy")

    UAV_trajectory_dueling_9 = UAV_trajectory_dueling_9[:15000, :, :]
    UE_Schedulings_dueling_9 = UE_Schedulings_dueling_9[:15000, :]
    Task_offloading_dueling_9 = Task_offloading_dueling_9[:15000, :, :]
    UAV_flight_time_dueling_9 = UAV_flight_time_dueling_9[:15000, :]
    Task_offloading_dueling_9 = Task_offloading_dueling_9[:15000, :, :]
    Convergence_duelin_9 = Convergence_dueling_9[:15000, :]
    Slot_dueling_9 = Slot_dueling_9[:15000, :]


    plot_energy_efficiency_avg(UAV_trajectory_dueling_5, UE_Schedulings_dueling_5,
                           Task_offloading_dueling_5,
                           UAV_flight_time_dueling_5,
                           Task_offloading_time_dueling_5, Slot_dueling_5, UAV_trajectory_dueling_9,
                           UE_Schedulings_dueling_9,
                           Task_offloading_dueling_9,
                           UAV_flight_time_dueling_9,
                           Task_offloading_time_dueling_9, Slot_dueling_9,
                           14999)
