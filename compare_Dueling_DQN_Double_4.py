import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare

env = UAV_MEC()
res = Res_plot_compare(env)
if __name__ == '__main__':
    # DQN
    # 用于生成吞吐量，能耗
    UAV_trajectory_dqn = np.load("Modify_DQN_Double_Dueling_2/UAV_trajectory_dqn.npy")
    UE_Schedulings_dqn = np.load("Modify_DQN_Double_Dueling_2/UE_Schedulings_dqn.npy")
    Task_offloading_dqn = np.load("Modify_DQN_Double_Dueling_2/Task_offloading_dqn.npy")
    UAV_flight_time_dqn = np.load("Modify_DQN_Double_Dueling_2/UAV_flight_time_dqn.npy")
    Task_offloading_time_dqn = np.load("Modify_DQN_Double_Dueling_2/Task_offloading_time_dqn.npy")
    Convergence_dqn = np.load("Modify_DQN_Double_Dueling_2/Convergence_dqn.npy")

    UAV_trajectory_dueling = np.load("Modify_DQN_Double_Dueling_2/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling = np.load("Modify_DQN_Double_Dueling_2/UE_Schedulings_dueling.npy")
    Task_offloading_dueling = np.load("Modify_DQN_Double_Dueling_2/Task_offloading_dueling.npy")
    UAV_flight_time_dueling = np.load("Modify_DQN_Double_Dueling_2/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling = np.load("Modify_DQN_Double_Dueling_2/Task_offloading_time_dueling.npy")
    Convergence_dueling = np.load("Modify_DQN_Double_Dueling_2/Convergence_dueling.npy")

    # Double
    UAV_trajectory_double = np.load("Modify_DQN_Double_Dueling_2/UAV_trajectory_double.npy")
    UE_Schedulings_double = np.load("Modify_DQN_Double_Dueling_2/UE_Schedulings_double.npy")
    Task_offloading_double = np.load("Modify_DQN_Double_Dueling_2/Task_offloading_double.npy")
    UAV_flight_time_double = np.load("Modify_DQN_Double_Dueling_2/UAV_flight_time_double.npy")
    Task_offloading_time_double = np.load("Modify_DQN_Double_Dueling_2/Task_offloading_time_double.npy")
    Convergence_double = np.load("Modify_DQN_Double_Dueling_2/Convergence_double.npy")


    res.plot_propulsion_energy(UAV_trajectory_dqn, UAV_trajectory_double, UAV_trajectory_dueling,
                           UAV_flight_time_dqn, UAV_flight_time_double, UAV_flight_time_dueling, 1999)

    res.plot_data_throughput(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
                         UAV_flight_time_dueling,
                         UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
                         Task_offloading_double,
                         Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
                         UE_Schedulings_double,
                         Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double, 1999)
    #
    res.plot_energy_efficiency(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
                             UAV_flight_time_dueling,
                             UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
                             Task_offloading_double,
                             Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
                             UE_Schedulings_double,
                             Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double, 1999)
