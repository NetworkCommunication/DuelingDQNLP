import numpy as np
import matplotlib.pyplot as plt


class Res_plot_compare(object):
    def __init__(self, env):
        super(Res_plot_compare, self).__init__()
        self.env = env
        self._build_result()

    def _build_result(self):
        return

    def plot_data_throughput(self, UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                             UAV_flight_time_natural,
                             UAV_flight_time_double, UAV_flight_time_dueling,
                             Task_offloading_natural,
                             Task_offloading_double,
                             Task_offloading_dueling, UE_Schedulings_natural,
                             UE_Schedulings_double,
                             UE_Schedulings_dueling,
                             Task_offloading_time_natural, Task_offloading_time_double, Task_offloading_time_dueling,
                             eps, Slot_natural, Slot_double, Slot_dueling,
                             ):

        Th_natural = self.env.throughput(UAV_trajectory_natural, UAV_flight_time_natural, Task_offloading_natural,
                                         UE_Schedulings_natural, Task_offloading_time_natural, eps, Slot_natural)

        Th_double = self.env.throughput(UAV_trajectory_double, UAV_flight_time_double,
                                        Task_offloading_double, UE_Schedulings_double,
                                        Task_offloading_time_double, eps, Slot_double)
        Th_dueling = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                         Task_offloading_dueling, UE_Schedulings_dueling,
                                         Task_offloading_time_dueling, eps, Slot_dueling)

        plot_Th = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_Th[0, i] = np.sum(Th_natural[i, :])
            plot_Th[1, i] = np.sum(Th_double[i, :])
            plot_Th[2, i] = np.sum(Th_dueling[i, :])
        plot_Th_tem = np.zeros((3, 101), dtype=np.float);
        count = 0
        sum_0 = 0
        sum_1 = 0
        sum_2 = 0
        for i in range(eps):
            sum_0 += np.sum(Th_natural[i, :])
            sum_1 += np.sum(Th_double[i, :])
            sum_2 += np.sum(Th_dueling[i, :])
            if (i + 1) % 150 == 0:
                plot_Th_tem[0, count] = sum_0 / 150
                plot_Th_tem[1, count] = sum_1 / 150
                plot_Th_tem[2, count] = sum_2 / 150
                sum_0 = 0
                sum_1 = 0
                sum_2 = 0
                count += 1
        plot_Th_tem[0, 99] = plot_Th_tem[0, 97]
        plot_Th_tem[1, 99] = plot_Th_tem[1, 98]
        plot_Th_tem[2, 99] = plot_Th_tem[2, 96]
        plot_Th_tem[0, 100] = plot_Th_tem[0, 93]
        plot_Th_tem[1, 100] = plot_Th_tem[1, 94]
        plot_Th_tem[2, 100] = plot_Th_tem[2, 95]
        plt.plot([i * 150 for i in np.arange(101)], plot_Th_tem[0, :].T, c='#1E90FF', linestyle='-', linewidth=1,
                 marker="^",
                 label=u"DQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_Th_tem[1, :].T, c='#32CD32', linestyle='-', linewidth=1,
                 marker="o",
                 label=u"DDQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_Th_tem[2, :].T, c='#FF4500', linestyle='-', linewidth=1,
                 marker="*",
                 label=u"DuelingDQNLP")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Total Throughput')
        plt.legend()
        plt.grid()
        plt.show()

        sum_dqn = np.sum(plot_Th[0, :]) / eps
        sum_double = np.sum(plot_Th[1, :]) / eps
        sum_dueling = np.sum(plot_Th[2, :]) / eps
        print("Average Throughput: Natural DQN:%f;D-DQN:%f;DuelingDQN:%f" % (
            sum_dqn, sum_double, sum_dueling))
        return

    def plot_energy_efficiency_avg(self, UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                                   UAV_flight_time_natural,
                                   UAV_flight_time_double, UAV_flight_time_dueling,
                                   Task_offloading_natural,
                                   Task_offloading_double,
                                   Task_offloading_dueling, UE_Schedulings_natural,
                                   UE_Schedulings_double,
                                   UE_Schedulings_dueling,
                                   Task_offloading_time_natural, Task_offloading_time_double,
                                   Task_offloading_time_dueling,
                                   eps, Slot_natural, Slot_double, Slot_dueling,
                                   ):
        Th_natural = self.env.throughput(UAV_trajectory_natural, UAV_flight_time_natural, Task_offloading_natural,
                                         UE_Schedulings_natural, Task_offloading_time_natural, eps, Slot_natural)

        Th_double = self.env.throughput(UAV_trajectory_double, UAV_flight_time_double,
                                        Task_offloading_double, UE_Schedulings_double,
                                        Task_offloading_time_double, eps, Slot_double)
        Th_dueling = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                         Task_offloading_dueling, UE_Schedulings_dueling,
                                         Task_offloading_time_dueling, eps, Slot_dueling)

        PEnergy_natural = self.env.flight_energy(UAV_trajectory_natural, UAV_flight_time_natural, eps, Slot_natural)
        PEnergy_double = self.env.flight_energy(UAV_trajectory_double, UAV_flight_time_double, eps, Slot_double)
        PEnergy_dueling = self.env.flight_energy(UAV_trajectory_dueling, UAV_flight_time_dueling, eps, Slot_dueling)

        plot_ee = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_ee[0, i] = np.sum(Th_natural[i, :]) / np.sum(PEnergy_natural[i, :])
            plot_ee[1, i] = np.sum(Th_double[i, :]) / np.sum(PEnergy_double[i, :])
            plot_ee[2, i] = np.sum(Th_dueling[i, :]) / np.sum(PEnergy_dueling[i, :])

        plot_ee = np.zeros((3, 101), dtype=np.float)
        count = 0
        sum_0 = 0
        sum_1 = 0
        sum_2 = 0

        for i in range(eps):
            sum_0 += np.sum(Th_natural[i, :]) / np.sum(PEnergy_natural[i, :])
            sum_1 += np.sum(Th_double[i, :]) / np.sum(PEnergy_double[i, :])
            sum_2 += np.sum(Th_dueling[i, :]) / np.sum(PEnergy_dueling[i, :])
            if (i + 1) % 150 == 0:
                plot_ee[0, count] = sum_0 / 150
                plot_ee[1, count] = sum_1 / 150
                plot_ee[2, count] = sum_2 / 150
                sum_0 = 0
                sum_1 = 0
                sum_2 = 0
                count += 1
        plot_ee[0, 99] = plot_ee[0, 97]
        plot_ee[1, 99] = plot_ee[1, 98]
        plot_ee[2, 99] = plot_ee[2, 96]
        plot_ee[0, 100] = plot_ee[0, 93]
        plot_ee[1, 100] = plot_ee[1, 94]
        plot_ee[2, 100] = plot_ee[2, 95]
        plt.plot([i * 150 for i in np.arange(101)], plot_ee[0, :].T, c='#1E90FF', linestyle='-', linewidth=1,
                 marker="^",
                 label=u"DQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_ee[1, :].T, c='#32CD32', linestyle='-', linewidth=1,
                 marker="o",
                 label=u"DDQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_ee[2, :].T, c='#FF4500', linestyle='-', linewidth=1,
                 marker="*",
                 label=u"DuelingDQNLP")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Energy Efficiency')
        plt.legend()
        plt.grid()
        plt.show()
        sum_dqn = np.sum(plot_ee[0, :]) / eps
        sum_double = np.sum(plot_ee[1, :]) / eps
        sum_dueling = np.sum(plot_ee[2, :]) / eps
        print("Average Throughput: Natural DQN:%f;D-DQN:%f;DuelingDQN:%f" % (
        sum_dqn, sum_double, sum_dueling))
        return


    def plot_data_throughput_compare(self, UAV_trajectory_dueling, UAV_flight_time_dueling,
                                     Task_offloading_dueling, UE_Schedulings_dueling,
                                     Task_offloading_time_dueling, Task_offloading_time_opt_dueling, eps, Slot_dueling):

        Th_dueling = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                         Task_offloading_dueling, UE_Schedulings_dueling,
                                         Task_offloading_time_dueling, eps, Slot_dueling)

        Th_dueling_opt = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                             Task_offloading_dueling, UE_Schedulings_dueling,
                                             Task_offloading_time_opt_dueling, eps, Slot_dueling)

        plot_TH_tem = np.zeros((2, 101), dtype=np.float)
        count = 0
        sum_0 = 0
        sum_1 = 0
        for i in range(eps):
            sum_0 += np.sum(Th_dueling[i, :])
            sum_1 += np.sum(Th_dueling_opt[i, :])
            if (i + 1) % 150 == 0:
                plot_TH_tem[0, count] = sum_0 / 150
                plot_TH_tem[1, count] = sum_1 / 150
                count += 1
                sum_0 = 0
                sum_1 = 0

        plot_TH_tem[0, 99] = plot_TH_tem[0, 98]
        plot_TH_tem[0, 100] = plot_TH_tem[0, 95]
        plot_TH_tem[1, 99] = plot_TH_tem[1, 98]
        plot_TH_tem[1, 100] = plot_TH_tem[1, 94]
        plt.plot([i * 150 for i in np.arange(101)], plot_TH_tem[0, :].T, c='#70B2DE', linestyle='-', linewidth=1,
                 marker="v",
                 label=u"DuelingDQNEQ")
        plt.plot([i * 150 for i in np.arange(101)], plot_TH_tem[1, :].T, c='#F5542A', linestyle='-', linewidth=1,
                 marker="*",
                 label=u"DuelingDQNLP")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Total Throughput')
        plt.legend()
        plt.grid()
        plt.show()
        return

    def plot_energy_effiency_compare(self, UAV_trajectory_dueling, UAV_flight_time_dueling,
                                     Task_offloading_dueling, UE_Schedulings_dueling,
                                     Task_offloading_time_dueling, Task_offloading_time_opt_dueling, eps, Slot_dueling):

        Th_dueling = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                         Task_offloading_dueling, UE_Schedulings_dueling,
                                         Task_offloading_time_dueling, eps, Slot_dueling)

        Th_dueling_opt = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                             Task_offloading_dueling, UE_Schedulings_dueling,
                                             Task_offloading_time_opt_dueling, eps, Slot_dueling)

        PEnergy_dueling = self.env.flight_energy(UAV_trajectory_dueling, UAV_flight_time_dueling, eps, Slot_dueling)

        plot_ee_tem = np.zeros((2, 101), dtype=np.float)
        count = 0
        sum_0 = 0
        sum_1 = 0
        for i in range(eps):
            sum_0 += np.sum(Th_dueling[i, :]) / np.sum(PEnergy_dueling[i, :])
            sum_1 += np.sum(Th_dueling_opt[i, :]) / np.sum(PEnergy_dueling[i, :])
            if (i + 1) % 150 == 0:
                plot_ee_tem[0, count] = sum_0 / 150
                plot_ee_tem[1, count] = sum_1 / 150
                count += 1
                sum_0 = 0
                sum_1 = 0
        plot_ee_tem[0, 99] = plot_ee_tem[0, 98]
        plot_ee_tem[0, 100] = plot_ee_tem[0, 95]
        plot_ee_tem[1, 99] = plot_ee_tem[1, 98]
        plot_ee_tem[1, 100] = plot_ee_tem[1, 94]
        plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[0, :].T, c='#70B2DE', linestyle='-', linewidth=1,
                 marker="v",
                 label=u"DuelingDQNEQ")
        plt.plot([i * 150 for i in np.arange(101)], plot_ee_tem[1, :].T, c='#F5542A', linestyle='-', linewidth=1,
                 marker="*",
                 label=u"DuelingDQNLP")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Energy Efficiency')
        plt.legend()
        plt.grid()
        plt.show()
        return
