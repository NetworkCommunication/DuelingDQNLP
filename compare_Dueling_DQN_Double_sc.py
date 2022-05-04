'''
Batch size值的比较
注意生成的.npy文件的存放路径
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import matplotlib.pyplot as plt

env = UAV_MEC()
res = Res_plot_compare(env)

def plot_energy_efficiency_avg(UAV_trajectory_dueling_32, UE_Schedulings_dueling_32,
                           Task_offloading_dueling_32,
                           UAV_flight_time_dueling_32,
                           Task_offloading_time_dueling_32, Slot_dueling_32, UAV_trajectory_dueling_64,
                           UE_Schedulings_dueling_64,
                           Task_offloading_dueling_64,
                           UAV_flight_time_dueling_64,
                           Task_offloading_time_dueling_64, Slot_dueling_64, UAV_trajectory_dueling_128,
                           UE_Schedulings_dueling_128,
                           Task_offloading_dueling_128,
                           UAV_flight_time_dueling_128,
                           Task_offloading_time_dueling_128, Slot_dueling_128, UAV_trajectory_dueling_256,
                           UE_Schedulings_dueling_256,
                           Task_offloading_dueling_256,
                           UAV_flight_time_dueling_256,
                           Task_offloading_time_dueling_256, Slot_dueling_256,
                           eps):
    Th_32 = env.throughput(UAV_trajectory_dueling_32, UAV_flight_time_dueling_32,
                             Task_offloading_dueling_32, UE_Schedulings_dueling_32,
                             Task_offloading_time_dueling_32, eps, Slot_dueling_32)
    Th_64 = env.throughput(UAV_trajectory_dueling_64, UAV_flight_time_dueling_64, Task_offloading_dueling_64,
                             UE_Schedulings_dueling_64, Task_offloading_time_dueling_64, eps, Slot_dueling_64)
    Th_128 = env.throughput(UAV_trajectory_dueling_128, UAV_flight_time_dueling_128, Task_offloading_dueling_128,
                             UE_Schedulings_dueling_128, Task_offloading_time_dueling_128, eps, Slot_dueling_128)
    Th_256 = env.throughput(UAV_trajectory_dueling_256, UAV_flight_time_dueling_256,
                              Task_offloading_dueling_256,
                              UE_Schedulings_dueling_256, Task_offloading_time_dueling_256, eps, Slot_dueling_256)

    PEnergy_32 = env.flight_energy(UAV_trajectory_dueling_32, UAV_flight_time_dueling_32, eps, Slot_dueling_32)
    PEnergy_64 = env.flight_energy(UAV_trajectory_dueling_64, UAV_flight_time_dueling_64, eps, Slot_dueling_64)
    PEnergy_128 = env.flight_energy(UAV_trajectory_dueling_128, UAV_flight_time_dueling_128, eps, Slot_dueling_128)
    PEnergy_256 = env.flight_energy(UAV_trajectory_dueling_256, UAV_flight_time_dueling_256, eps,
                                      Slot_dueling_256)

    plot_ee = np.zeros((4, 101), dtype=np.float64)
    plot_ee_tem = np.zeros((4, 101), dtype=np.float64);
    count = 0
    sum_0 = 0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0

    for i in range(eps):
        sum_0 += np.sum(Th_32[i, :]) / np.sum(PEnergy_32[i, :])
        sum_1 += np.sum(Th_64[i, :]) / np.sum(PEnergy_64[i, :])
        sum_2 += np.sum(Th_128[i, :]) / np.sum(PEnergy_128[i, :])
        sum_3 += np.sum(Th_256[i, :]) / np.sum(PEnergy_256[i, :])
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
    plot_ee[1, 99] = plot_ee[0, 98]
    plot_ee[2, 99] = plot_ee[0, 96]
    plot_ee[3, 99] = plot_ee[0, 97]
    plot_ee[0, 100]=plot_ee[0, 98]
    plot_ee[1, 100]=plot_ee[0, 94]
    plot_ee[2, 100]=plot_ee[0, 95]
    plot_ee[3, 100]=plot_ee[0, 98]
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[0, :].T, c='k', linestyle='-', linewidth=1,
             label=u"Batch size=32", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[1, :].T, c='g', linestyle='-', linewidth=1,
             label=u"Batch size=64", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[2, :].T, c='r', linestyle='-', linewidth=1,
             label=u"Batch size=128", alpha=0.8)
    plt.plot([i * 150 for i in np.arange(101)], plot_ee[3, :].T, c='b', linestyle='-', linewidth=1,
             label=u"Batch size=256", alpha=0.8)
    plt.ylim((0.5, 0.9))
    plt.xlabel(u'Episode')
    plt.ylabel(u'Energy Efficiency')
    plt.legend()
    plt.grid()
    plt.show()
    sum_dqn = np.sum(plot_ee[0, :]) / eps
    sum_double = np.sum(plot_ee[1, :]) / eps
    sum_dueling = np.sum(plot_ee[2, :]) / eps
    sum_double_dueling = np.sum(plot_ee[3, :]) / eps
    print("Average Throughput: Natural DQN:%f;D-DQN:%f;DuelingDQN:%f;DDuelingDQN:%f" % (
    sum_dqn, sum_double, sum_dueling, sum_double_dueling))
    return


if __name__ == '__main__':


    UAV_trajectory_dueling_32 = np.load("Modify_Batch_size/32/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_32 = np.load("Modify_Batch_size/32/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_32 = np.load("Modify_Batch_size/32/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_32 = np.load("Modify_Batch_size/32/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_32 = np.load("Modify_Batch_size/32/Task_offloading_time_dueling.npy")
    Convergence_dueling_32 = np.load("Modify_Batch_size/32/Convergence_dueling.npy")
    Slot_dueling_32 = np.load("Modify_Batch_size/32/Slot_dueling.npy")

    UAV_trajectory_dueling_64 = np.load("Modify_Batch_size/64/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_64 = np.load("Modify_Batch_size/64/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_64 = np.load("Modify_Batch_size/64/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_64 = np.load("Modify_Batch_size/64/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_64 = np.load("Modify_Batch_size/64/Task_offloading_time_dueling.npy")
    Convergence_dueling_64 = np.load("Modify_Batch_size/64/Convergence_dueling.npy")
    Slot_dueling_64 = np.load("Modify_Batch_size/64/Slot_dueling.npy")


    UAV_trajectory_dueling_128 = np.load("Modify_Batch_size/128/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_128 = np.load("Modify_Batch_size/128/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_128 = np.load("Modify_Batch_size/128/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_128 = np.load("Modify_Batch_size/128/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_128 = np.load("Modify_Batch_size/128/Task_offloading_time_dueling.npy")
    Convergence_dueling_128 = np.load("Modify_Batch_size/128/Convergence_dueling.npy")
    Slot_dueling_128 = np.load("Modify_Batch_size/128/Slot_dueling.npy")


    UAV_trajectory_dueling_256 = np.load("Modify_Batch_size/256/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling_256 = np.load("Modify_Batch_size/256/UE_Schedulings_dueling.npy")
    Task_offloading_dueling_256 = np.load("Modify_Batch_size/256/Task_offloading_dueling.npy")
    UAV_flight_time_dueling_256 = np.load("Modify_Batch_size/256/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling_256 = np.load("Modify_Batch_size/256/Task_offloading_time_dueling.npy")
    Convergence_dueling_256 = np.load("Modify_Batch_size/256/Convergence_dueling.npy")
    Slot_dueling_256 = np.load("Modify_Batch_size/256/Slot_dueling.npy")

    plot_energy_efficiency_avg(UAV_trajectory_dueling_32, UE_Schedulings_dueling_32,
                               Task_offloading_dueling_32,
                               UAV_flight_time_dueling_32,
                               Task_offloading_time_dueling_32, Slot_dueling_32, UAV_trajectory_dueling_64,
                               UE_Schedulings_dueling_64,
                               Task_offloading_dueling_64,
                               UAV_flight_time_dueling_64,
                               Task_offloading_time_dueling_64, Slot_dueling_64, UAV_trajectory_dueling_128,
                               UE_Schedulings_dueling_128,
                               Task_offloading_dueling_128,
                               UAV_flight_time_dueling_128,
                               Task_offloading_time_dueling_128, Slot_dueling_128, UAV_trajectory_dueling_256,
                               UE_Schedulings_dueling_256,
                               Task_offloading_dueling_256,
                               UAV_flight_time_dueling_256,
                               Task_offloading_time_dueling_256, Slot_dueling_256,
                               14999)
