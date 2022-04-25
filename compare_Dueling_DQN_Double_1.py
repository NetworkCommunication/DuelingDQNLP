import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare

env = UAV_MEC()
res = Res_plot_compare(env)
if __name__ == '__main__':
    # UAV_trajectory_dqn = np.load("Num_UE_10/UAV_trajectory_dqn.npy")
    # UE_Schedulings_dqn = np.load("Num_UE_10/UE_Schedulings_dqn.npy")
    # Task_offloading_dqn = np.load("Num_UE_10/Task_offloading_dqn.npy")
    # UAV_flight_time_dqn = np.load("Num_UE_10/UAV_flight_time_dqn.npy")
    # Task_offloading_time_dqn = np.load("Num_UE_10/Task_offloading_time_dqn.npy")
    # Convergence_dqn = np.load("Num_UE_10/Convergence_dqn.npy")
    #
    # UAV_trajectory_double = np.load("Num_UE_10/UAV_trajectory_dueling.npy")
    # UE_Schedulings_double = np.load("Num_UE_10/UE_Schedulings_dueling.npy")
    # Task_offloading_double = np.load("Num_UE_10/Task_offloading_dueling.npy")
    # UAV_flight_time_double = np.load("Num_UE_10/UAV_flight_time_dueling.npy")
    # Task_offloading_time_double = np.load("Num_UE_10/Task_offloading_time_dueling.npy")
    # Convergence_double = np.load("Num_UE_10/Convergence_dueling.npy")
    #
    # UAV_trajectory_dueling = np.load("Num_UE_10/UAV_trajectory_dueling.npy")
    # UE_Schedulings_dueling = np.load("Num_UE_10/UE_Schedulings_dueling.npy")
    # Task_offloading_dueling = np.load("Num_UE_10/Task_offloading_dueling.npy")
    # UAV_flight_time_dueling = np.load("Num_UE_10/UAV_flight_time_dueling.npy")
    # Task_offloading_time_dueling = np.load("Num_UE_10/Task_offloading_time_dueling.npy")
    # Convergence_dueling = np.load("Num_UE_10/Convergence_dueling.npy")

    UAV_trajectory_dueling_cvx = np.load("CVX(1)/UAV_trajectory_dqn_cvx.npy")
    UE_Schedulings_dueling_cvx = np.load("CVX(1)/UE_Schedulings_dqn_cvx.npy")
    Task_offloading_dueling_cvx = np.load("CVX(1)/Task_offloading_dqn_cvx.npy")
    UAV_flight_time_dueling_cvx = np.load("CVX(1)/UAV_flight_time_dqn_cvx.npy")
    Task_offloading_time_dueling_cvx = np.load("CVX(1)/Task_offloading_time_dqn_cvx.npy")
    Convergence_dueling_cvx = np.load("CVX(1)/Convergence_dqn_cvx_1.npy")

    UAV_trajectory_dueling = np.load("Dueling(4)/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling = np.load("Dueling(4)/UE_Schedulings_dueling.npy")
    Task_offloading_dueling = np.load("Dueling(4)/Task_offloading_dueling.npy")
    UAV_flight_time_dueling = np.load("Dueling(4)/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling = np.load("Dueling(4)/Task_offloading_time_dueling.npy")
    Convergence_dueling = np.load("Dueling(4)/Convergence_dueling.npy")


    res.plot_propulsion_energy(UAV_trajectory_dueling_cvx, UAV_trajectory_dueling_cvx, UAV_trajectory_dueling,
                           UAV_flight_time_dueling_cvx, UAV_flight_time_dueling_cvx, UAV_flight_time_dueling, 1999)

    # res.plot_data_throughput(UAV_trajectory_dueling, UAV_trajectory_dueling_cvx, UAV_trajectory_dueling_cvx,
    #                      UAV_flight_time_dueling,
    #                      UAV_flight_time_dueling_cvx, UAV_flight_time_dueling_cvx, Task_offloading_dueling,
    #                      Task_offloading_dueling_cvx,
    #                      Task_offloading_dueling_cvx, UE_Schedulings_dueling, UE_Schedulings_dueling_cvx,
    #                      UE_Schedulings_dueling_cvx,
    #                      Task_offloading_time_dueling, Task_offloading_time_dueling_cvx, Task_offloading_time_dueling_cvx, 1999)

    # res.plot_energy_efficiency(UAV_trajectory_dueling, UAV_trajectory_dueling_cvx, UAV_trajectory_dueling_cvx,
    #                      UAV_flight_time_dueling,
    #                      UAV_flight_time_dueling_cvx, UAV_flight_time_dueling_cvx, Task_offloading_dueling,
    #                      Task_offloading_dueling_cvx,
    #                      Task_offloading_dueling_cvx, UE_Schedulings_dueling, UE_Schedulings_dueling_cvx,
    #                      UE_Schedulings_dueling_cvx,
    #                      Task_offloading_time_dueling, Task_offloading_time_dueling_cvx, Task_offloading_time_dueling_cvx, 1999)
    #
