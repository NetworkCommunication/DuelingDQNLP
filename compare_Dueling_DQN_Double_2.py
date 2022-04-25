import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import matplotlib.pyplot as plt

env = UAV_MEC()
res = Res_plot_compare(env)

#比较的是修改模型的索引

def plot_propulsion_energy(UAV_trajectory_dqn, UAV_trajectory_double, UAV_trajectory_dueling,
                           UAV_flight_time_dqn, UAV_flight_time_double, UAV_flight_time_dueling, eps):
    PEnergy_dqn = env.flight_energy(UAV_trajectory_dqn, UAV_flight_time_dqn, eps)
    PEnergy_double = env.flight_energy(UAV_trajectory_double, UAV_flight_time_double, eps)
    PEnergy_dueling = env.flight_energy(UAV_trajectory_dueling, UAV_flight_time_dueling, eps)

    plot_energy = np.zeros((3, eps), dtype=np.float)
    for i in range(eps):
        plot_energy[0, i] = np.sum(PEnergy_dqn[i, :])
        plot_energy[1, i] = np.sum(PEnergy_double[i, :])
        plot_energy[2, i] = np.sum(PEnergy_dueling[i, :])
    plt.plot(np.arange(eps), plot_energy[0, :].T, c='k', linestyle='-', marker='*',
             linewidth=1,
             label=u"DLP")
    plt.plot(np.arange(eps), plot_energy[1, :].T.T, c='b', linestyle='-', marker='*',
             linewidth=1,
             label=u"DDLP")
    plt.plot(np.arange(eps), plot_energy[2, :].T, c='r', linestyle='-', marker='o',
             linewidth=1,
             label=u"DRLLP")
    plt.xlabel(u'Episode')
    plt.ylabel(u'Energy Consumption')
    plt.legend()
    plt.grid()
    plt.show()
    return


def plot_data_throughput(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
                         UAV_flight_time_dueling,
                         UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
                         Task_offloading_double,
                         Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
                         UE_Schedulings_double,
                         Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double, eps):
    Th_dqn = env.throughput(UAV_trajectory_dqn, UAV_flight_time_dqn, Task_offloading_dqn,
                            UE_Schedulings_dqn, Task_offloading_time_dqn, eps)

    Th_double = env.throughput(UAV_trajectory_double, UAV_flight_time_double,
                               Task_offloading_double, UE_Schedulings_double,
                               Task_offloading_time_double, eps)
    Th_dueling = env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                Task_offloading_dueling, UE_Schedulings_dueling,
                                Task_offloading_time_dueling, eps)

    plot_Th = np.zeros((3, eps), dtype=np.float)
    for i in range(eps):
        plot_Th[0, i] = np.sum(Th_dqn[i, :])
        plot_Th[1, i] = np.sum(Th_double[i, :])
        plot_Th[2, i] = np.sum(Th_dueling[i, :])
    plt.plot(np.arange(eps), plot_Th[0, :].T, c='k', linestyle='-', marker='*', linewidth=1,
             label=u"DLP")
    plt.plot(np.arange(eps), plot_Th[1, :].T, c='b', linestyle='-', marker='*', linewidth=1,
             label=u"DDLP")
    plt.plot(np.arange(eps), plot_Th[2, :].T, c='r', linestyle='-', marker='o', linewidth=1,
             label=u"DRLLP")
    plt.xlabel(u'Episode')
    plt.ylabel(u'Total Throughput')
    plt.legend()
    plt.grid()
    plt.show()
    return


def plot_energy_efficiency(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
                           UAV_flight_time_dueling,
                           UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
                           Task_offloading_double,
                           Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
                           UE_Schedulings_double,
                           Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double,
                           eps):
    Th_dqn = env.throughput(UAV_trajectory_dqn, UAV_flight_time_dqn, Task_offloading_dqn,
                            UE_Schedulings_dqn, Task_offloading_time_dqn, eps)

    Th_double = env.throughput(UAV_trajectory_double, UAV_flight_time_double,
                               Task_offloading_double, UE_Schedulings_double,
                               Task_offloading_time_double, eps)
    Th_dueling = env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                Task_offloading_dueling, UE_Schedulings_dueling,
                                Task_offloading_time_dueling, eps)

    PEnergy_dqn = env.flight_energy(UAV_trajectory_dqn, UAV_flight_time_dqn, eps)
    PEnergy_double = env.flight_energy(UAV_trajectory_double, UAV_flight_time_double, eps)
    PEnergy_dueling = env.flight_energy(UAV_trajectory_dueling, UAV_flight_time_dueling, eps)

    plot_ee = np.zeros((3, eps), dtype=np.float)
    for i in range(eps):
        plot_ee[0, i] = np.sum(Th_dqn[i, :]) / np.sum(PEnergy_dqn[i, :])
        plot_ee[1, i] = np.sum(Th_double[i, :]) / np.sum(PEnergy_double[i, :])
        plot_ee[2, i] = np.sum(Th_dueling[i, :]) / np.sum(PEnergy_dueling[i, :])
    plt.plot(np.arange(eps), plot_ee[0, :].T, c='k', linestyle='-', marker='<', linewidth=1,
             label=u"DLP")
    plt.plot(np.arange(eps), plot_ee[1, :].T, c='b', linestyle='-', marker='*', linewidth=1,
             label=u"DDLP")
    plt.plot(np.arange(eps), plot_ee[2, :].T, c='r', linestyle='-', marker='o', linewidth=1,
             label=u"DRLLP")
    plt.xlabel(u'Episode')
    plt.ylabel(u'Energy Efficiency')
    plt.legend()
    plt.grid()
    plt.show()
    return


if __name__ == '__main__':
    UAV_trajectory_dqn = np.load("Modify_DQN_Double_Dueling_1/UAV_trajectory_dqn.npy")
    UE_Schedulings_dqn = np.load("Modify_DQN_Double_Dueling_1/UE_Schedulings_dqn.npy")
    Task_offloading_dqn = np.load("Modify_DQN_Double_Dueling_1/Task_offloading_dqn.npy")
    UAV_flight_time_dqn = np.load("Modify_DQN_Double_Dueling_1/UAV_flight_time_dqn.npy")
    Task_offloading_time_dqn = np.load("Modify_DQN_Double_Dueling_1/Task_offloading_time_dqn.npy")
    Convergence_dqn = np.load("Modify_DQN_Double_Dueling_1/Convergence_dqn.npy")

    UAV_trajectory_double = np.load("Modify_DQN_Double_Dueling_1/UAV_trajectory_double.npy")
    UE_Schedulings_double = np.load("Modify_DQN_Double_Dueling_1/UE_Schedulings_double.npy")
    Task_offloading_double = np.load("Modify_DQN_Double_Dueling_1/Task_offloading_double.npy")
    UAV_flight_time_double = np.load("Modify_DQN_Double_Dueling_1/UAV_flight_time_double.npy")
    Task_offloading_time_double = np.load("Modify_DQN_Double_Dueling_1/Task_offloading_time_double.npy")
    Convergence_double = np.load("Modify_DQN_Double_Dueling_1/Convergence_double.npy")

    UAV_trajectory_dueling = np.load("Modify_DQN_Double_Dueling_1/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling = np.load("Modify_DQN_Double_Dueling_1/UE_Schedulings_dueling.npy")
    Task_offloading_dueling = np.load("Modify_DQN_Double_Dueling_1/Task_offloading_dueling.npy")
    UAV_flight_time_dueling = np.load("Modify_DQN_Double_Dueling_1/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling = np.load("Modify_DQN_Double_Dueling_1/Task_offloading_time_dueling.npy")
    Convergence_dueling = np.load("Modify_DQN_Double_Dueling_1/Convergence_dueling.npy")

    plot_propulsion_energy(UAV_trajectory_dqn, UAV_trajectory_double, UAV_trajectory_dueling,
                           UAV_flight_time_dqn, UAV_flight_time_double, UAV_flight_time_dueling, 199)

    plot_data_throughput(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
                             UAV_flight_time_dueling,
                             UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
                             Task_offloading_double,
                             Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
                             UE_Schedulings_double,
                             Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double, 199)
    #
    plot_energy_efficiency(UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
                               UAV_flight_time_dueling,
                               UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
                               Task_offloading_double,
                               Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
                               UE_Schedulings_double,
                               Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double,
                               199)
