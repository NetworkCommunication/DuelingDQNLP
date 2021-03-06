'''
Learning rate值的比较
注意生成的.npy文件的存放路径
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import matplotlib.pyplot as plt

env = UAV_MEC()
res = Res_plot_compare(env)


def plot_energy_efficiency_avg(UAV_trajectory_dueling_0005, UE_Schedulings_dueling_0005,
                           Task_offloading_dueling_0005,
                           UAV_flight_time_dueling_0005,
                           Task_offloading_time_dueling_0005, Slot_dueling_0005, UAV_trajectory_dueling_001,
                           UE_Schedulings_dueling_001,
                           Task_offloading_dueling_001,
                           UAV_flight_time_dueling_001,
                           Task_offloading_time_dueling_001, Slot_dueling_001, UAV_trajectory_dueling_005,
                           UE_Schedulings_dueling_005,
                           Task_offloading_dueling_005,
                           UAV_flight_time_dueling_005,
                           Task_offloading_time_dueling_005, Slot_dueling_005, UAV_trajectory_dueling_01,
                           UE_Schedulings_dueling_01,
                           Task_offloading_dueling_01,
                           UAV_flight_time_dueling_01,
                           Task_offloading_time_dueling_01, Slot_dueling_01,
                           eps):
    Th_0005 = env.throughput(UAV_trajectory_dueling_0005, UAV_flight_time_dueling_0005,
                             Task_offloading_dueling_0005, UE_Schedulings_dueling_0005,
                             Task_offloading_time_dueling_0005, eps, Slot_dueling_0005)
    Th_001 = env.throughput(UAV_trajectory_dueling_001, UAV_flight_time_dueling_001, Task_offloading_dueling_001,
                             UE_Schedulings_dueling_001, Task_offloading_time_dueling_001, eps, Slot_dueling_001)
    Th_005 = env.throughput(UAV_trajectory_dueling_005, UAV_flight_time_dueling_005, Task_offloading_dueling_005,
                             UE_Schedulings_dueling_005, Task_offloading_time_dueling_005, eps, Slot_dueling_005)
    Th_01 = env.throughput(UAV_trajectory_dueling_01, UAV_flight_time_dueling_01,
                              Task_offloading_dueling_01,
                              UE_Schedulings_dueling_01, Task_offloading_time_dueling_01, eps, Slot_dueling_01)

    PEnergy_0005 = env.flight_energy(UAV_trajectory_dueling_0005, UAV_flight_time_dueling_0005, eps, Slot_dueling_0005)
    PEnergy_001 = env.flight_energy(UAV_trajectory_dueling_001, UAV_flight_time_dueling_001, eps, Slot_dueling_001)
    PEnergy_005 = env.flight_energy(UAV_trajectory_dueling_005, UAV_flight_time_dueling_005, eps, Slot_dueling_005)
    PEnergy_01 = env.flight_energy(UAV_trajectory_dueling_01, UAV_flight_time_dueling_01, eps,
                                      Slot_dueling_01)

    plot_ee = np.zeros((4, 101), dtype=np.float64)
    plot_ee_tem = np.zeros((4, 101), dtype=np.float64);
    count = 0
    sum_0 = 0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0

    for i in range(eps):
        sum_0 += np.sum(Th_0005[i, :]) / np.sum(PEnergy_0005[i, :])
        sum_1 += np.sum(Th_001[i, :]) / np.sum(PEnergy_001[i, :])
        sum_2 += np.sum(Th_005[i, :]) / np.sum(PEnergy_005[i, :])
        sum_3 += np.sum(Th_01[i, :]) / np.sum(PEnergy_01[i, :])
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
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[0, :].T, c='g', linestyle='-', linewidth=1,
             label=u"Learning rate=0.005", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[1, :].T, c='b', linestyle='-', linewidth=1,
             label=u"Learning rate=0.01", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[2, :].T, c='r', linestyle='-', linewidth=1,
             label=u"Learning rate=0.05", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[3, :].T, c='k', linestyle='-', linewidth=1,
             label=u"Learning rate=0.1", alpha=0.8)
    plt.ylim((0.5, 0.9))
    plt.xlabel(u'Episode')
    plt.ylabel(u'Energy Efficiency')
    plt.legend()
    plt.grid()
    plt.show()
    return


if __name__ == '__main__':
    UAV_trajectory_dueling_0005 = np.load("Modify_Learning_rate/0005/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_0005 = np.load("Modify_Learning_rate/0005/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_0005 = np.load("Modify_Learning_rate/0005/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_0005 = np.load("Modify_Learning_rate/0005/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_0005 = np.load("Modify_Learning_rate/0005/Task_offloading_time_dueling.npy")
    Convergence_dueling_0005 = np.load("Modify_Learning_rate/0005/Convergence_dueling.npy")
    Slot_dueling_0005 = np.load("Modify_Learning_rate/0005/Slot_dueling.npy")

    UAV_trajectory_dueling_001 = np.load("Modify_Learning_rate/001/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_001 = np.load("Modify_Learning_rate/001/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_001 = np.load("Modify_Learning_rate/001/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_001 = np.load("Modify_Learning_rate/001/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_001 = np.load("Modify_Learning_rate/001/Task_offloading_time_dueling.npy")
    Convergence_dueling_001 = np.load("Modify_Learning_rate/001/Convergence_dueling.npy")
    Slot_dueling_001 = np.load("Modify_Learning_rate/001/Slot_dueling.npy")

    UAV_trajectory_dueling_005 = np.load("Modify_Learning_rate/005/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_005 = np.load("Modify_Learning_rate/005/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_005 = np.load("Modify_Learning_rate/005/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_005 = np.load("Modify_Learning_rate/005/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_005 = np.load("Modify_Learning_rate/005/Task_offloading_time_dueling.npy")
    Convergence_dueling_005 = np.load("Modify_Learning_rate/005/Convergence_dueling.npy")
    Slot_dueling_005 = np.load("Modify_Learning_rate/005/Slot_dueling.npy")

    UAV_trajectory_dueling_01 = np.load("Modify_Learning_rate/01/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_01 = np.load("Modify_Learning_rate/01/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_01 = np.load("Modify_Learning_rate/01/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_01 = np.load("Modify_Learning_rate/01/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_01 = np.load("Modify_Learning_rate/01/Task_offloading_time_dueling.npy")
    Convergence_dueling_01 = np.load("Modify_Learning_rate/01/Convergence_dueling.npy")
    Slot_dueling_01 = np.load("Modify_Learning_rate/01/Slot_dueling.npy")

    plot_energy_efficiency_avg(UAV_trajectory_dueling_0005, UE_Schedulings_dueling_0005,
                           Task_offloading_dueling_0005,
                           UAV_flight_time_dueling_0005,
                           Task_offloading_time_dueling_0005, Slot_dueling_0005, UAV_trajectory_dueling_001,
                           UE_Schedulings_dueling_001,
                           Task_offloading_dueling_001,
                           UAV_flight_time_dueling_001,
                           Task_offloading_time_dueling_001, Slot_dueling_001, UAV_trajectory_dueling_005,
                           UE_Schedulings_dueling_005,
                           Task_offloading_dueling_005,
                           UAV_flight_time_dueling_005,
                           Task_offloading_time_dueling_005, Slot_dueling_005, UAV_trajectory_dueling_01,
                           UE_Schedulings_dueling_01,
                           Task_offloading_dueling_01,
                           UAV_flight_time_dueling_01,
                           Task_offloading_time_dueling_01, Slot_dueling_01,
                           14999)
