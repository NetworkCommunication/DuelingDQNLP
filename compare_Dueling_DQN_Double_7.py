'''
基于Dueling DQN 比较凸优化优化飞行时间和Duelng DQN优化飞行时间
问题：修改环境中的周期数

比较无人机的飞行速度
'''
import numpy as np
from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import matplotlib.pyplot as plt

env = UAV_MEC()
res = Res_plot_compare(env)


def plot_UAV_GT(w_k, UAV_trajectory_dqn, UAV_trajectory_double, UAV_trajectory_dueling):
    for e in range(1910, 1920):
        # UAV_trajectory_dqn[e, :] = env.UAV_FLY(UAV_trajectory_dqn[e, :])
        # UAV_trajectory_double[e, :] =env.UAV_FLY(UAV_trajectory_double[e, :])
        # UAV_trajectory_dueling[e, :] =env.UAV_FLY(UAV_trajectory_dueling[e, :])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_dqn = []
        y_dqn = []
        z_dqn = []

        x_double = []
        y_double = []
        z_double = []

        x_dueling = []
        y_dueling = []
        z_dueling = []

        for slot in range(env.N_slot):
            if UAV_trajectory_dqn[e, slot, 0]!=0:
                x_dqn.append(UAV_trajectory_dqn[e, slot, 0])
                y_dqn.append(UAV_trajectory_dqn[e, slot, 1])
                z_dqn.append(UAV_trajectory_dqn[e, slot, 2])
            if UAV_trajectory_double[e, slot, 0] != 0:
                x_double.append(UAV_trajectory_double[e, slot, 0])
                y_double.append(UAV_trajectory_double[e, slot, 1])
                z_double.append(UAV_trajectory_double[e, slot, 2])
            if UAV_trajectory_dueling[e, slot, 0] != 0:
                x_dueling.append(UAV_trajectory_dueling[e, slot, 0])
                y_dueling.append(UAV_trajectory_dueling[e, slot, 1])
                z_dueling.append(UAV_trajectory_dueling[e, slot, 2])

        ax.scatter(w_k[:, 0], w_k[:, 1], c='g', marker='o', label=u"TD locations")
        ax.plot(x_dqn[:], y_dqn[:], z_dqn[:], c='k', linestyle='-', marker='',
                label=u"DLP")
        ax.plot(x_double[:], y_double[:], z_double[:], c='b', linestyle='--', marker='',
                label=u"DDLP")
        ax.plot(x_dueling[:], y_dueling[:], z_dueling[:], c='r', linestyle='--', marker='',
                label=u"DRLLP")
        ax.set_zlim(0, 250)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plt.legend(prop=myfont)
        plt.legend()
        plt.show()

        plt.plot(x_dqn[:], y_dqn[:], c='k', linestyle='-', marker='', label=u"DLP")
        plt.plot(x_double[:], y_double[:], c='b', linestyle='--', marker='', label=u"DDLP")
        plt.plot(x_dueling[:], y_dueling[:], c='r', linestyle='--', marker='', label=u"DRLLP")
        plt.scatter(w_k[:, 0], w_k[:, 1], c='g', marker='o', label=u"GT Locations")
        plt.ylabel(u'x(m)')
        plt.xlabel(u'y(m)')
        plt.legend()
        plt.grid()
        plt.show()
    return


if __name__ == '__main__':
    UAV_trajectory_dqn = np.load("loc_DQN_Double_Dueling_E_3/UAV_trajectory_dqn.npy")
    UE_Schedulings_dqn = np.load("loc_DQN_Double_Dueling_E_3/UE_Schedulings_dqn.npy")
    Task_offloading_dqn = np.load("loc_DQN_Double_Dueling_E_3/Task_offloading_dqn.npy")
    UAV_flight_time_dqn = np.load("loc_DQN_Double_Dueling_E_3/UAV_flight_time_dqn.npy")
    Task_offloading_time_dqn = np.load("loc_DQN_Double_Dueling_E_3/Task_offloading_time_dqn.npy")
    Convergence_dqn = np.load("loc_DQN_Double_Dueling_E_3/Convergence_dqn.npy")

    # 用于生成吞吐量
    UAV_trajectory_dueling = np.load("loc_DQN_Double_Dueling_E_3/UAV_trajectory_dueling.npy")
    UE_Schedulings_dueling = np.load("loc_DQN_Double_Dueling_E_3/UE_Schedulings_dueling.npy")
    Task_offloading_dueling = np.load("loc_DQN_Double_Dueling_E_3/Task_offloading_dueling.npy")
    UAV_flight_time_dueling = np.load("loc_DQN_Double_Dueling_E_3/UAV_flight_time_dueling.npy")
    Task_offloading_time_dueling = np.load("loc_DQN_Double_Dueling_E_3/Task_offloading_time_dueling.npy")
    Convergence_dueling = np.load("loc_DQN_Double_Dueling_E_3/Convergence_dueling.npy")

    # Double
    # 用于生成能耗
    # 用于生成吞吐量
    UAV_trajectory_double = np.load("loc_DQN_Double_Dueling_E_3/UAV_trajectory_double.npy")
    UE_Schedulings_double = np.load("loc_DQN_Double_Dueling_E_3/UE_Schedulings_double.npy")
    Task_offloading_double = np.load("loc_DQN_Double_Dueling_E_3/Task_offloading_double.npy")
    UAV_flight_time_double = np.load("loc_DQN_Double_Dueling_E_3/UAV_flight_time_double.npy")
    Task_offloading_time_double = np.load("loc_DQN_Double_Dueling_E_3/Task_offloading_time_double.npy")
    Convergence_double = np.load("loc_DQN_Double_Dueling_E_3/Convergence_double.npy")

    plot_UAV_GT(env.w_k, UAV_trajectory_dqn, UAV_trajectory_double, UAV_trajectory_dueling)