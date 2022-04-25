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
    # Dueling
    # 用于生成能耗
    # Dueling
    # 用于生成能耗
    UAV_trajectory_dueling = np.load("DDD/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling = np.load("DDD/UE_Schedulings_dueling.npy")
    Task_offloading_dueling = np.load("DDD/Task_offloading_dueling.npy")
    UAV_flight_time_dueling = np.load("DDD/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling = np.load("DDD/Task_offloading_time_dueling.npy")
    Convergence_dueling = np.load("DDD/Convergence_dueling.npy")
    Slot_dueling = np.load("DDD/Slot_dueling.npy")
    Task_offloading_time_no_opt_dueling = np.load("DDD/Task_offloading_time_no_opt_dueling.npy")
    UAV_trajectory_dueling = UAV_trajectory_dueling[:15000, :, :]
    UE_Schedulings_dueling = UE_Schedulings_dueling[:15000, :]
    Task_offloading_dueling = Task_offloading_dueling[:15000, :, :]
    UAV_flight_time_dueling = UAV_flight_time_dueling[:15000, :]
    Task_offloading_time_dueling = Task_offloading_time_dueling[:15000, :, :]
    Convergence_dueling = Convergence_dueling[:15000, :]
    Slot_dueling = Slot_dueling[:15000, :]
    Task_offloading_time_no_opt_dueling = Task_offloading_time_no_opt_dueling[:15000, :]
    res.plot_data_throughput_compare(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                     Task_offloading_dueling, UE_Schedulings_dueling,
                                     Task_offloading_time_no_opt_dueling,
                                     Task_offloading_time_dueling, 14999, Slot_dueling)
    res.plot_energy_effiency_compare(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                     Task_offloading_dueling, UE_Schedulings_dueling,
                                     Task_offloading_time_no_opt_dueling, Task_offloading_time_dueling, 14999,
                                     Slot_dueling)
