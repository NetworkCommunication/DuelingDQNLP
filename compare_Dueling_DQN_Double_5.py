'''
基于Dueling DQN 比较凸优化优化飞行时间和Duelng DQN优化飞行时间
问题：修改环境中的周期数

比较2000轮次，每20个画一个。
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare

env = UAV_MEC()
res = Res_plot_compare(env)
if __name__ == '__main__':
    # DQN
    # 用于生成吞吐量，能耗
    UAV_trajectory_dqn = np.load("DQN(6)/UAV_trajectory_dqn.npy")
    UE_Schedulings_dqn = np.load("DQN(6)/UE_Schedulings_dqn.npy")
    Task_offloading_dqn = np.load("DQN(6)/Task_offloading_dqn.npy")
    UAV_flight_time_dqn = np.load("DQN(6)/UAV_flight_time_dqn.npy")
    Task_offloading_time_dqn = np.load("DQN(6)/Task_offloading_time_dqn.npy")
    Convergence_dqn = np.load("DQN(6)/Convergence_dqn.npy")

    # Dueling
    # 用于生成能耗
    # UAV_trajectory_dueling = np.load("Double_Dueling/UAV_trajectory_dueling.npy")
    # UE_Schedulings_dueling = np.load("Double_Dueling/UE_Schedulings_dueling.npy")
    # Task_offloading_dueling = np.load("Double_Dueling/Task_offloading_dueling.npy")
    # UAV_flight_time_dueling = np.load("Double_Dueling/UAV_flight_time_dueling.npy")
    # Task_offloading_time_dueling = np.load("Double_Dueling/Task_offloading_time_dueling.npy")
    # Convergence_dueling = np.load("Double_Dueling/Convergence_dueling.npy")

    # 修改第100和第1480 eposide
    # 用于生成能耗
    # UAV_trajectory_dueling = np.load("DQN_Dueling/UAV_trajectory_dueling.npy")
    # UE_Schedulings_dueling = np.load("DQN_Dueling/UE_Schedulings_dueling.npy")
    # Task_offloading_dueling = np.load("DQN_Dueling/Task_offloading_dueling.npy")
    # UAV_flight_time_dueling = np.load("DQN_Dueling/UAV_flight_time_dueling.npy")
    # Task_offloading_time_dueling = np.load("DQN_Dueling/Task_offloading_time_dueling.npy")
    # Convergence_dueling = np.load("DQN_Dueling/Convergence_dueling.npy")

    # 用于生成吞吐量
    UAV_trajectory_dueling = np.load("Dueling(4)/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling = np.load("Dueling(4)/UE_Schedulings_dueling.npy")
    Task_offloading_dueling = np.load("Dueling(4)/Task_offloading_dueling.npy")
    UAV_flight_time_dueling = np.load("Dueling(4)/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling = np.load("Dueling(4)/Task_offloading_time_dueling.npy")
    Convergence_dueling = np.load("Dueling(4)/Convergence_dueling.npy")
    # 时间未优化
    Task_offloading_time_no_dueling = np.load("Time_compare/Task_offloading_time_dueling.npy")

    # Double
    # 用于生成能耗
    # 用于生成吞吐量
    UAV_trajectory_double = np.load("Double_Dueling/UAV_trajectory_double.npy")
    UE_Schedulings_double = np.load("Double_Dueling/UE_Schedulings_double.npy")
    Task_offloading_double = np.load("Double_Dueling/Task_offloading_double.npy")
    UAV_flight_time_double = np.load("Double_Dueling/UAV_flight_time_double.npy")
    Task_offloading_time_double = np.load("Double_Dueling/Task_offloading_time_double.npy")
    Convergence_double = np.load("Double_Dueling/Convergence_double.npy")

    UAV_trajectory_double = UAV_trajectory_double[:2000, :, :]
    UE_Schedulings_double = UE_Schedulings_double[:2000, :, :]
    Task_offloading_double = Task_offloading_double[:2000, :, :]
    UAV_flight_time_double = UAV_flight_time_double[:2000, :]
    Task_offloading_double = Task_offloading_double[:2000, :, :]
    Convergence_double = Convergence_double[:2000, :]
    # res.plot_propulsion_energy(UAV_trajectory_dqn, UAV_trajectory_double, UAV_trajectory_dueling,
    #                        UAV_flight_time_dqn, UAV_flight_time_double, UAV_flight_time_dueling, 1999)

    # res.plot_data_throughput(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
    #                      UAV_flight_time_dueling,
    #                      UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
    #                      Task_offloading_double,
    #                      Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
    #                      UE_Schedulings_double,
    #                      Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double, 1999)
    #
    # res.plot_energy_efficiency(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
    #                          UAV_flight_time_dueling,
    #                          UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
    #                          Task_offloading_double,
    #                          Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
    #                          UE_Schedulings_double,
    #                          Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double, 1999)

    # res.plot_data_throughput_compare(UAV_trajectory_dueling,
    #                                  UAV_flight_time_dueling,
    #                                  Task_offloading_dueling, UE_Schedulings_dueling
    #                                  , Task_offloading_time_no_dueling, Task_offloading_time_dueling, 1999)

    res.plot_UAV_GT(env.w_k, UAV_trajectory_dqn, UAV_trajectory_double, UAV_trajectory_dueling)