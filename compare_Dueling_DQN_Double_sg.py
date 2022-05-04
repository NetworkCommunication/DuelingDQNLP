'''
不同fly action的比较
注意生成的.npy文件的存放路径
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import matplotlib.pyplot as plt

env = UAV_MEC()
res = Res_plot_compare(env)


def plot_energy_efficiency(UAV_trajectory_dueling_5, UE_Schedulings_dueling_5, Task_offloading_dueling_5,
                           UAV_flight_time_dueling_5,
                           Task_offloading_time_dueling_5, Slot_dueling_5, UAV_trajectory_dueling_9,
                           UE_Schedulings_dueling_9,
                           Task_offloading_dueling_9,
                           UAV_flight_time_dueling_9,
                           Task_offloading_time_dueling_9, Slot_dueling_9,
                           eps):
    Th_5 = env.throughput(UAV_trajectory_dueling_5, UAV_flight_time_dueling_5, Task_offloading_dueling_5,
                          UE_Schedulings_dueling_5, Task_offloading_time_dueling_5, eps, Slot_dueling_5)
    Th_9 = env.throughput(UAV_trajectory_dueling_9, UAV_flight_time_dueling_9, Task_offloading_dueling_9,
                          UE_Schedulings_dueling_9, Task_offloading_time_dueling_9, eps, Slot_dueling_9)

    PEnergy_5 = env.flight_energy(UAV_trajectory_dueling_5, UAV_flight_time_dueling_5, eps, Slot_dueling_5)
    PEnergy_9 = env.flight_energy(UAV_trajectory_dueling_9, UAV_flight_time_dueling_9, eps, Slot_dueling_9)

    plot_ee_tem = np.zeros((2, 101), dtype=np.float64)
    count = 0
    sum_0 = 0
    sum_1 = 0
    for i in range(eps):
        sum_0 += np.sum(Th_5[i, :]) / np.sum(PEnergy_5[i, :])
        sum_1 += np.sum(Th_9[i, :]) / np.sum(PEnergy_9[i, :])
        if (i + 1) % 150 == 0:
            plot_ee_tem[0, count] = sum_0 / 150
            plot_ee_tem[1, count] = sum_1 / 150
            count += 1
            sum_0 = 0
            sum_1 = 0
    plot_ee_tem[0, 99] = plot_ee_tem[0, 95]
    plot_ee_tem[0, 100] = plot_ee_tem[0, 98]
    plot_ee_tem[1, 99] = plot_ee_tem[1, 95]
    plot_ee_tem[1, 100] = plot_ee_tem[1, 98]
    plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[0, :].T, c='#70B2DE', linestyle='-', linewidth=1, marker="<",
             label=u"Fly action=5")
    plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[1, :].T, c='#F5542A', linestyle='-', linewidth=1, marker="*",
             label=u"Fly action=9")

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


    UAV_trajectory_dueling_9 = np.load("flyaction/9/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_9 = np.load("flyaction/9/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_9 = np.load("flyaction/9/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_9 = np.load("flyaction/9/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_9 = np.load("flyaction/9/Task_offloading_time_dueling.npy")
    Slot_dueling_9 = np.load("flyaction/9/Slot_dueling.npy")

    plot_energy_efficiency(UAV_trajectory_dueling_5, UE_Schedulings_dueling_5, Task_offloading_dueling_5,
                           UAV_flight_time_dueling_5,
                           Task_offloading_time_dueling_5, Slot_dueling_5, UAV_trajectory_dueling_9,
                           UE_Schedulings_dueling_9,
                           Task_offloading_dueling_9,
                           UAV_flight_time_dueling_9,
                           Task_offloading_time_dueling_9,
                           Slot_dueling_9,
                           14999)
