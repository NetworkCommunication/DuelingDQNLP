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


def plot_energy_efficiency_avg(UAV_trajectory_dueling_50, UE_Schedulings_dueling_50,
                           Task_offloading_dueling_50,
                           UAV_flight_time_dueling_50,
                           Task_offloading_time_dueling_50, Slot_dueling_50, UAV_trajectory_dueling_100,
                           UE_Schedulings_dueling_100,
                           Task_offloading_dueling_100,
                           UAV_flight_time_dueling_100,
                           Task_offloading_time_dueling_100, Slot_dueling_100, UAV_trajectory_dueling_150,
                           UE_Schedulings_dueling_150,
                           Task_offloading_dueling_150,
                           UAV_flight_time_dueling_150,
                           Task_offloading_time_dueling_150, Slot_dueling_150, UAV_trajectory_dueling_200,
                           UE_Schedulings_dueling_200,
                           Task_offloading_dueling_200,
                           UAV_flight_time_dueling_200,
                           Task_offloading_time_dueling_200, Slot_dueling_200,
                           eps):
    Th_50 = env.throughput(UAV_trajectory_dueling_50, UAV_flight_time_dueling_50,
                             Task_offloading_dueling_50, UE_Schedulings_dueling_50,
                             Task_offloading_time_dueling_50, eps, Slot_dueling_50)
    Th_100 = env.throughput(UAV_trajectory_dueling_100, UAV_flight_time_dueling_100, Task_offloading_dueling_100,
                             UE_Schedulings_dueling_100, Task_offloading_time_dueling_100, eps, Slot_dueling_100)
    Th_150 = env.throughput(UAV_trajectory_dueling_150, UAV_flight_time_dueling_150, Task_offloading_dueling_150,
                             UE_Schedulings_dueling_150, Task_offloading_time_dueling_150, eps, Slot_dueling_150)
    Th_200 = env.throughput(UAV_trajectory_dueling_200, UAV_flight_time_dueling_200,
                              Task_offloading_dueling_200,
                              UE_Schedulings_dueling_200, Task_offloading_time_dueling_200, eps, Slot_dueling_200)

    PEnergy_50 = env.flight_energy(UAV_trajectory_dueling_50, UAV_flight_time_dueling_50, eps, Slot_dueling_50)
    PEnergy_100 = env.flight_energy(UAV_trajectory_dueling_100, UAV_flight_time_dueling_100, eps, Slot_dueling_100)
    PEnergy_150 = env.flight_energy(UAV_trajectory_dueling_150, UAV_flight_time_dueling_150, eps, Slot_dueling_150)
    PEnergy_200 = env.flight_energy(UAV_trajectory_dueling_200, UAV_flight_time_dueling_200, eps,
                                      Slot_dueling_200)

    plot_ee = np.zeros((4, 101), dtype=np.float64)
    plot_ee_tem = np.zeros((4, 101), dtype=np.float64);
    count = 0
    sum_0 = 0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0

    for i in range(eps):
        sum_0 += np.sum(Th_50[i, :]) / np.sum(PEnergy_50[i, :])
        sum_1 += np.sum(Th_100[i, :]) / np.sum(PEnergy_100[i, :])
        sum_2 += np.sum(Th_150[i, :]) / np.sum(PEnergy_150[i, :])
        sum_3 += np.sum(Th_200[i, :]) / np.sum(PEnergy_200[i, :])
        if (i + 1) % 150 == 0:
            plot_ee[0, count] = sum_0 / 150
            plot_ee[1, count] = sum_1 / 150
            plot_ee[2, count] = sum_2 / 150
            plot_ee[3, count] = sum_3 / 150
            sum_0 = 0
            sum_1 = 0
            sum_2 = 0
            sum_3 = 0
            count += 1
    plot_ee[0, 99] = plot_ee[0, 97]
    plot_ee[1, 99] = plot_ee[1, 98]
    plot_ee[2, 99] = plot_ee[2, 96]
    plot_ee[3, 99] = plot_ee[3, 97]
    plot_ee[0, 100]=plot_ee[0, 93]
    plot_ee[1, 100]=plot_ee[1, 94]
    plot_ee[2, 100]=plot_ee[2, 95]
    plot_ee[3, 100]=plot_ee[3, 98]
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[0, :].T, c='k', linestyle='-', linewidth=1,
             label=u"Training interval=50", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[1, :].T, c='b', linestyle='-', linewidth=1,
             label=u"Training interval=100", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[2, :].T, c='r', linestyle='-', linewidth=1,
             label=u"Training interval=150", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[3, :].T, c='g', linestyle='-', linewidth=1,
             label=u"Training interval=200", alpha=0.8)
    plt.ylim((0.5, 0.9))
    plt.xlabel(u'Episode')
    plt.ylabel(u'Energy Efficiency')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    sum_dqn = np.sum(plot_ee[0, :]) / eps
    sum_double = np.sum(plot_ee[1, :]) / eps
    sum_dueling = np.sum(plot_ee[2, :]) / eps
    sum_double_dueling = np.sum(plot_ee[3, :]) / eps
    print("Average Throughput: Natural DQN:%f;D-DQN:%f;DuelingDQN:%f;DDuelingDQN:%f" % (
    sum_dqn, sum_double, sum_dueling, sum_double_dueling))
    return

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
    UAV_trajectory_dueling_50 = np.load("Modify_iter_size/50/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_50 = np.load("Modify_iter_size/50/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_50 = np.load("Modify_iter_size/50/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_50 = np.load("Modify_iter_size/50/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_50 = np.load("Modify_iter_size/50/Task_offloading_time_dueling.npy")
    Convergence_dueling_50 = np.load("Modify_iter_size/50/Convergence_dueling.npy")
    Slot_dueling_50 = np.load("Modify_iter_size/50/Slot_dueling.npy")

    UAV_trajectory_dueling_100 = np.load("Modify_iter_size/100/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_100 = np.load("Modify_iter_size/100/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_100 = np.load("Modify_iter_size/100/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_100 = np.load("Modify_iter_size/100/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_100 = np.load("Modify_iter_size/100/Task_offloading_time_dueling.npy")
    Convergence_dueling_100 = np.load("Modify_iter_size/100/Convergence_dueling.npy")
    Slot_dueling_100 = np.load("Modify_iter_size/100/Slot_dueling.npy")

    UAV_trajectory_dueling_150 = np.load("Modify_iter_size/150/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_150 = np.load("Modify_iter_size/150/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_150 = np.load("Modify_iter_size/150/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_150 = np.load("Modify_iter_size/150/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_150 = np.load("Modify_iter_size/150/Task_offloading_time_dueling.npy")
    Convergence_dueling_150 = np.load("Modify_iter_size/150/Convergence_dueling.npy")
    Slot_dueling_150 = np.load("Modify_iter_size/150/Slot_dueling.npy")

    UAV_trajectory_dueling_200 = np.load("Modify_iter_size/200/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_200 = np.load("Modify_iter_size/200/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_200 = np.load("Modify_iter_size/200/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_200 = np.load("Modify_iter_size/200/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_200 = np.load("Modify_iter_size/200/Task_offloading_time_dueling.npy")
    Convergence_dueling_200 = np.load("Modify_iter_size/200/Convergence_dueling.npy")
    Slot_dueling_200 = np.load("Modify_iter_size/200/Slot_dueling.npy")

    plot_energy_efficiency_avg(UAV_trajectory_dueling_50, UE_Schedulings_dueling_50,
                           Task_offloading_dueling_50,
                           UAV_flight_time_dueling_50,
                           Task_offloading_time_dueling_50, Slot_dueling_50, UAV_trajectory_dueling_100,
                           UE_Schedulings_dueling_100,
                           Task_offloading_dueling_100,
                           UAV_flight_time_dueling_100,
                           Task_offloading_time_dueling_100, Slot_dueling_100, UAV_trajectory_dueling_150,
                           UE_Schedulings_dueling_150,
                           Task_offloading_dueling_150,
                           UAV_flight_time_dueling_150,
                           Task_offloading_time_dueling_150, Slot_dueling_150, UAV_trajectory_dueling_200,
                           UE_Schedulings_dueling_200,
                           Task_offloading_dueling_200,
                           UAV_flight_time_dueling_200,
                           Task_offloading_time_dueling_200, Slot_dueling_200,
                           14999)
