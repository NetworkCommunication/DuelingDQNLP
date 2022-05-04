'''
比较DQNLP,DDQNLP,DUelingDQNLP
注意生成的.npy文件的存放路径
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare

env = UAV_MEC()
res = Res_plot_compare(env)
if __name__ == '__main__':
    # DQN
    # 用于生成吞吐量，能耗
    UAV_trajectory_natural = np.load("DDD/UAV_trajectory_natural.npy")
    UE_Schedulings_natural = np.load("DDD/UE_Schedulings_natural.npy")
    Task_offloading_natural = np.load("DDD/Task_offloading_natural.npy")
    UAV_flight_time_natural = np.load("DDD/UAV_flight_time_natural.npy")
    Task_offloading_time_natural = np.load("DDD/Task_offloading_time_natural.npy")
    Convergence_natural = np.load("DDD/Convergence_natural.npy")
    Slot_natural = np.load("DDD/Slot_natural.npy")

    UAV_trajectory_natural = UAV_trajectory_natural[:15000, :, :]
    UE_Schedulings_natural = UE_Schedulings_natural[:15000, :]
    Task_offloading_natural = Task_offloading_natural[:15000, :, :]
    UAV_flight_time_natural = UAV_flight_time_natural[:15000, :]
    Task_offloading_natural = Task_offloading_natural[:15000, :, :]
    Convergence_natural = Convergence_natural[:15000, :]
    Slot_natural = Slot_natural[:15000, :]

    # Dueling
    # 用于生成能耗
    UAV_trajectory_dueling = np.load("DDD/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling = np.load("DDD/UE_Schedulings_dueling.npy")
    Task_offloading_dueling = np.load("DDD/Task_offloading_dueling.npy")
    UAV_flight_time_dueling = np.load("DDD/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling = np.load("DDD/Task_offloading_time_dueling.npy")
    Convergence_dueling = np.load("DDD/Convergence_dueling.npy")
    Slot_dueling = np.load("DDD/Slot_dueling.npy")

    UAV_trajectory_dueling = UAV_trajectory_dueling[:15000, :, :]
    UE_Schedulings_dueling = UE_Schedulings_dueling[:15000, :]
    Task_offloading_dueling = Task_offloading_dueling[:15000, :, :]
    UAV_flight_time_dueling = UAV_flight_time_dueling[:15000, :]
    Task_offloading_dueling = Task_offloading_dueling[:15000, :, :]
    Convergence_dueling = Convergence_dueling[:15000, :]
    Slot_dueling = Slot_dueling[:15000, :]

    # Double
    # 用于生成能耗
    # 用于生成吞吐量
    UAV_trajectory_double = np.load("DDD/UAV_trajectory_double.npy")
    UE_Schedulings_double = np.load("DDD/UE_Schedulings_double.npy")
    Task_offloading_double = np.load("DDD/Task_offloading_double.npy")
    UAV_flight_time_double = np.load("DDD/UAV_flight_time_double.npy")
    Task_offloading_time_double = np.load("DDD/Task_offloading_time_double.npy")
    Convergence_double = np.load("DDD/Convergence_double.npy")
    Slot_double = np.load("DDD/Slot_double.npy")

    UAV_trajectory_double = UAV_trajectory_double[:15000, :, :]
    UE_Schedulings_double = UE_Schedulings_double[:15000, :]
    Task_offloading_double = Task_offloading_double[:15000, :, :]
    UAV_flight_time_double = UAV_flight_time_double[:15000, :]
    Task_offloading_double = Task_offloading_double[:15000, :, :]
    Convergence_double = Convergence_double[:15000, :]
    Slot_double = Slot_double[:15000, :]

    # 吞吐量图
    res.plot_data_throughput(UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                             UAV_flight_time_natural,
                             UAV_flight_time_double, UAV_flight_time_dueling,
                             Task_offloading_natural,
                             Task_offloading_double,
                             Task_offloading_dueling, UE_Schedulings_natural,
                             UE_Schedulings_double,
                             UE_Schedulings_dueling,
                             Task_offloading_time_natural, Task_offloading_time_double, Task_offloading_time_dueling,
                             14999, Slot_natural, Slot_double, Slot_dueling,
                             )

    # 能效图
    res.plot_energy_efficiency_avg(UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                                   UAV_flight_time_natural,
                                   UAV_flight_time_double, UAV_flight_time_dueling,
                                   Task_offloading_natural,
                                   Task_offloading_double,
                                   Task_offloading_dueling, UE_Schedulings_natural,
                                   UE_Schedulings_double,
                                   UE_Schedulings_dueling,
                                   Task_offloading_time_natural, Task_offloading_time_double,
                                   Task_offloading_time_dueling,
                                   14999, Slot_natural, Slot_double, Slot_dueling,
                                   )
