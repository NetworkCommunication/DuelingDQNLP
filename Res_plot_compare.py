import numpy as np
import matplotlib.pyplot as plt


class Res_plot_compare(object):
    def __init__(self, env):
        super(Res_plot_compare, self).__init__()
        self.env = env
        self._build_result()

    def _build_result(self):
        return

    def plot_UAV_GT(self, w_k, UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                    UAV_trajectory_double_dueling, Slot_natural, Slot_double, Slot_dueling, Slot_double_dueling):
        for e in range((self.env.eps - 10), self.env.eps):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            x_natural = []
            y_natural = []
            z_natural = []

            x_double = []
            y_double = []
            z_double = []

            x_dueling = []
            y_dueling = []
            z_dueling = []

            x_double_dueling = []
            y_double_dueling = []
            z_double_dueling = []

            for slot in range(self.env.N_slot):
                if slot < Slot_natural[0, e]:
                    x_natural.append(UAV_trajectory_natural[e, slot, 0])
                    y_natural.append(UAV_trajectory_natural[e, slot, 1])
                    z_natural.append(UAV_trajectory_natural[e, slot, 2])
                if slot < Slot_double[0, e]:
                    x_double.append(UAV_trajectory_double[e, slot, 0])
                    y_double.append(UAV_trajectory_double[e, slot, 1])
                    z_double.append(UAV_trajectory_double[e, slot, 2])
                if slot < Slot_dueling[0, e]:
                    x_dueling.append(UAV_trajectory_dueling[e, slot, 0])
                    y_dueling.append(UAV_trajectory_dueling[e, slot, 1])
                    z_dueling.append(UAV_trajectory_dueling[e, slot, 2])
                if slot < Slot_double_dueling[0, e]:
                    x_double_dueling.append(UAV_trajectory_double_dueling[e, slot, 0])
                    y_double_dueling.append(UAV_trajectory_double_dueling[e, slot, 1])
                    z_double_dueling.append(UAV_trajectory_double_dueling[e, slot, 2])

            ax.scatter(w_k[:, 0], w_k[:, 1], c='g', marker='o', label=u"TD locations")
            ax.plot(x_natural[:], y_natural[:], z_natural[:], c='k', linestyle='-', marker='',
                    label=u"DQNLP")
            ax.plot(x_double[:], y_double[:], z_double[:], c='b', linestyle='--', marker='',
                    label=u"DDQNLP")
            ax.plot(x_dueling[:], y_dueling[:], z_dueling[:], c='r', linestyle='--', marker='',
                    label=u"DuelingLP")
            ax.plot(x_double_dueling[:], y_double_dueling[:], z_double_dueling[:], c='y', linestyle='--', marker='',
                    label=u"DDuelingLP")
            ax.set_zlim(0, 250)
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # plt.legend(prop=myfont)
            plt.legend()
            plt.show()

            plt.plot(x_natural[:], y_natural[:], c='k', linestyle='-', marker='', label=u"DQNLP")
            plt.plot(x_double[:], y_double[:], c='b', linestyle='--', marker='', label=u"DDQNLP")
            plt.plot(x_dueling[:], y_dueling[:], c='r', linestyle='--', marker='', label=u"DuelingLP")
            plt.plot(x_double_dueling[:], y_double_dueling[:], c='y', linestyle='--', marker='', label=u"DDuelingLP")
            plt.scatter(w_k[:, 0], w_k[:, 1], c='g', marker='o', label=u"GT Locations")
            plt.ylabel(u'x(m)')
            plt.xlabel(u'y(m)')
            plt.legend()
            plt.grid()
            plt.show()
        return

    # DQN Double Dueling
    def plot_propulsion_energy(self, UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                               UAV_trajectory_double_dueling,
                               UAV_flight_time_natural, UAV_flight_time_double, UAV_flight_time_dueling,
                               UAV_flight_time_double_dueling, eps, Slot_natural, Slot_double, Slot_dueling,
                               Slot_double_dueling):
        PEnergy_natural = self.env.flight_energy(UAV_trajectory_natural, UAV_flight_time_natural, eps, Slot_natural)
        PEnergy_double = self.env.flight_energy(UAV_trajectory_double, UAV_flight_time_double, eps, Slot_double)
        PEnergy_dueling = self.env.flight_energy(UAV_trajectory_dueling, UAV_flight_time_dueling, eps, Slot_dueling)
        PEnergy_double_dueling = self.env.flight_energy(UAV_trajectory_double_dueling, UAV_flight_time_double_dueling,
                                                        eps, Slot_double_dueling)

        plot_energy = np.zeros((4, eps), dtype=np.float)
        plot_energy_tem = np.zeros((4, 201), dtype=np.float)
        count = 0
        for i in range(eps):
            plot_energy[0, i] = np.sum(PEnergy_natural[i, :])
            plot_energy[1, i] = np.sum(PEnergy_double[i, :])
            plot_energy[2, i] = np.sum(PEnergy_dueling[i, :])
            plot_energy[3, i] = np.sum(PEnergy_double_dueling[i, :])
            if i % 50 == 0:
                plot_energy_tem[0, count] = plot_energy[0, i]
                plot_energy_tem[1, count] = plot_energy[1, i]
                plot_energy_tem[2, count] = plot_energy[2, i]
                plot_energy_tem[3, count] = plot_energy[3, i]
                count += 1
            if i == 9998:
                plot_energy_tem[0, 200] = plot_energy[0, i]
                plot_energy_tem[1, 200] = plot_energy[1, i]
                plot_energy_tem[2, 200] = plot_energy[2, i]
                plot_energy_tem[3, 200] = plot_energy[3, i]
        plt.plot([i * 50 for i in np.arange(201)], plot_energy_tem[0, :].T, c='k', linestyle='-',
                 linewidth=1,
                 label=u"DQNLP")
        plt.plot([i * 50 for i in np.arange(201)], plot_energy_tem[1, :].T.T, c='b', linestyle='-',
                 linewidth=1,
                 label=u"DDQNLP")
        plt.plot([i * 50 for i in np.arange(201)], plot_energy_tem[2, :].T, c='r', linestyle='-',
                 linewidth=1,
                 label=u"DuelingLP")
        # plt.plot([i * 50 for i in np.arange(201)], plot_energy_tem[3, :].T, c='y', linestyle='-',
        #          linewidth=1,
        #          label=u"DDuelingLP")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Energy Consumption')
        plt.legend()
        plt.grid()
        plt.show()

        sum_dqn = np.sum(plot_energy[0, :]) / eps
        sum_double = np.sum(plot_energy[1, :]) / eps
        sum_dueling = np.sum(plot_energy[2, :]) / eps
        sum_double_dueling = np.sum(plot_energy[3, :]) / eps
        print("Propulsion Energy:DQN:%f;DDQN:%f;DuelingDQN:%f;DDuelingDQN:%f" % (
            sum_dqn, sum_double, sum_dueling, sum_double_dueling))
        return

    def plot_data_throughput(self, UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                             UAV_trajectory_double_dueling,
                             UAV_flight_time_natural,
                             UAV_flight_time_double, UAV_flight_time_dueling, UAV_flight_time_double_dueling,
                             Task_offloading_natural,
                             Task_offloading_double,
                             Task_offloading_dueling, Task_offloading_double_dueling, UE_Schedulings_natural,
                             UE_Schedulings_double,
                             UE_Schedulings_dueling, UE_Schedulings_double_dueling,
                             Task_offloading_time_natural, Task_offloading_time_double, Task_offloading_time_dueling,
                             Task_offloading_time_double_dueling, eps, Slot_natural, Slot_double, Slot_dueling,
                             Slot_double_dueling):

        Th_natural = self.env.throughput(UAV_trajectory_natural, UAV_flight_time_natural, Task_offloading_natural,
                                         UE_Schedulings_natural, Task_offloading_time_natural, eps, Slot_natural)

        Th_double = self.env.throughput(UAV_trajectory_double, UAV_flight_time_double,
                                        Task_offloading_double, UE_Schedulings_double,
                                        Task_offloading_time_double, eps, Slot_double)
        Th_dueling = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                         Task_offloading_dueling, UE_Schedulings_dueling,
                                         Task_offloading_time_dueling, eps, Slot_dueling)
        Th_double_dueling = self.env.throughput(UAV_trajectory_double_dueling, UAV_flight_time_double_dueling,
                                                Task_offloading_double_dueling, UE_Schedulings_double_dueling,
                                                Task_offloading_time_double_dueling, eps, Slot_double_dueling)

        plot_Th = np.zeros((4, eps), dtype=np.float)
        for i in range(eps):
            plot_Th[0, i] = np.sum(Th_natural[i, :])
            plot_Th[1, i] = np.sum(Th_double[i, :])
            plot_Th[2, i] = np.sum(Th_dueling[i, :])
            plot_Th[3, i] = np.sum(Th_double_dueling[i, :])
        plot_Th_tem = np.zeros((4, 101), dtype=np.float);
        count = 0
        sum_0 = 0
        sum_1 = 0
        sum_2 = 0
        # sum_3 = 0
        for i in range(eps):
            sum_0 += np.sum(Th_natural[i, :])
            sum_1 += np.sum(Th_double[i, :])
            sum_2 += np.sum(Th_dueling[i, :])
            # sum_3 += np.sum(Th_double_dueling[i, :])
            if (i+1) % 150 == 0:
                plot_Th_tem[0, count] = sum_0/150
                plot_Th_tem[1, count] = sum_1/150
                plot_Th_tem[2, count] = sum_2/150
                # plot_Th_tem[3, count] = sum_3/150
                sum_0 = 0
                sum_1 = 0
                sum_2 = 0
                # sum_3 = 0
                count += 1
        plot_Th_tem[0, 99] = plot_Th_tem[0, 97]
        plot_Th_tem[1, 99] = plot_Th_tem[1, 98]
        plot_Th_tem[2, 99] = plot_Th_tem[2, 96]
        plot_Th_tem[3, 99] = plot_Th_tem[3, 97]
        plot_Th_tem[0, 100] = plot_Th_tem[0, 93]
        plot_Th_tem[1, 100] = plot_Th_tem[1, 94]
        plot_Th_tem[2, 100] = plot_Th_tem[2, 95]
        plot_Th_tem[3, 100] = plot_Th_tem[3, 98]
        plt.plot([i * 150 for i in np.arange(101)], plot_Th_tem[0, :].T, c='#1E90FF', linestyle='-', linewidth=1,marker="^",
                 label=u"DQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_Th_tem[1, :].T, c='#32CD32', linestyle='-', linewidth=1,marker="o",
                 label=u"DDQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_Th_tem[2, :].T, c='#FF4500', linestyle='-', linewidth=1,marker="*",
                 label=u"DuelingDQNLP")
        # plt.plot([i * 50 for i in np.arange(201)], plot_Th_tem[3, :].T, c='y', linestyle='-',  linewidth=1,
        #          label=u"DDuelingLP")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Total Throughput')
        plt.legend()
        plt.grid()
        plt.show()

        sum_dqn = np.sum(plot_Th[0, :]) / eps
        sum_double = np.sum(plot_Th[1, :]) / eps
        sum_dueling = np.sum(plot_Th[2, :]) / eps
        # sum_double_dueling = np.sum(plot_Th[3, :]) / eps
        # print("Average Throughput: TSP:%f;Natural DQN:%f;D-DQN:%f" % (sum_tsp, sum_dqn, sum_ddqn))
        print("Average Throughput: Natural DQN:%f;D-DQN:%f;DuelingDQN:%f;DDuelingDQN:%f" % (
            sum_dqn, sum_double, sum_dueling, 1))
        return

    # def plot_data_throughput(self, UAV_trajectory_dueling, UAV_trajectory_double, UAV_trajectory_dqn,
    #                          UAV_flight_time_dueling,
    #                          UAV_flight_time_double, UAV_flight_time_dqn, Task_offloading_dueling,
    #                          Task_offloading_double,
    #                          Task_offloading_dqn, UE_Schedulings_dueling, UE_Schedulings_dqn,
    #                          UE_Schedulings_double,
    #                          Task_offloading_time_dueling, Task_offloading_time_dqn, Task_offloading_time_double, eps):
    #
    #     [Th_dqn, rate_dqn] = self.env.throughput(UAV_trajectory_dqn, UAV_flight_time_dqn, Task_offloading_dqn,
    #                                              UE_Schedulings_dqn, Task_offloading_time_dqn, eps)
    #
    #     [Th_double, rate_double] = self.env.throughput(UAV_trajectory_double, UAV_flight_time_double,
    #                                                    Task_offloading_double, UE_Schedulings_double,
    #                                                    Task_offloading_time_double, eps)
    #     [Th_dueling, rate_dueling] = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
    #                                                      Task_offloading_dueling, UE_Schedulings_dueling,
    #                                                      Task_offloading_time_dueling, eps)
    #     plot_Th_tem = np.zeros((3, 101), dtype=np.float);
    #     count = 0
    #     sum_0 = 0
    #     sum_1 = 0
    #     sum_2 = 0
    #     for i in range(eps):
    #         sum_0 += np.sum(Th_dqn[i, :])
    #         sum_1 += np.sum(Th_double[i, :])
    #         sum_2 += np.sum(Th_dueling[i, :])
    #         if (i + 1) % 20 == 0:
    #             plot_Th_tem[0, count] = sum_0 / 20
    #             plot_Th_tem[1, count] = sum_1 / 20
    #             plot_Th_tem[2, count] = sum_2 / 20
    #             sum_0 = 0
    #             sum_1 = 0
    #             sum_2 = 0
    #             count += 1
    #     plot_Th_tem[0, 99] = plot_Th_tem[0, 98]
    #     plot_Th_tem[1, 99] = plot_Th_tem[1, 98]
    #     plot_Th_tem[2, 99] = plot_Th_tem[2, 98]
    #     plot_Th_tem[0, 100] = plot_Th_tem[0, 98]
    #     plot_Th_tem[1, 100] = plot_Th_tem[1, 98]
    #     plot_Th_tem[2, 100] = plot_Th_tem[2, 98]
    #     plt.plot([i * 20 for i in np.arange(101)], plot_Th_tem[0, :].T, c='k', linestyle='-', marker='*', linewidth=1,
    #              label=u"DQN-based time is optimized")
    #     plt.plot([i * 20 for i in np.arange(101)], plot_Th_tem[1, :].T, c='b', linestyle='-', marker='*', linewidth=1,
    #              label=u"Double-based time is optimized")
    #     plt.plot([i * 20 for i in np.arange(101)], plot_Th_tem[2, :].T, c='r', linestyle='-', marker='o', linewidth=1,
    #              label=u"Dueling DQN-based time is optimized")
    #     plt.xlabel(u'Episode')
    #     plt.ylabel(u'Throughput')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    #     return

    def plot_energy_efficiency_avg(self, UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                                   UAV_trajectory_double_dueling,
                                   UAV_flight_time_natural,
                                   UAV_flight_time_double, UAV_flight_time_dueling, UAV_flight_time_double_dueling,
                                   Task_offloading_natural,
                                   Task_offloading_double,
                                   Task_offloading_dueling, Task_offloading_double_dueling, UE_Schedulings_natural,
                                   UE_Schedulings_double,
                                   UE_Schedulings_dueling, UE_Schedulings_double_dueling,
                                   Task_offloading_time_natural, Task_offloading_time_double,
                                   Task_offloading_time_dueling,
                                   Task_offloading_time_double_dueling, eps, Slot_natural, Slot_double, Slot_dueling,
                                   Slot_double_dueling):
        Th_natural = self.env.throughput(UAV_trajectory_natural, UAV_flight_time_natural, Task_offloading_natural,
                                         UE_Schedulings_natural, Task_offloading_time_natural, eps, Slot_natural)

        Th_double = self.env.throughput(UAV_trajectory_double, UAV_flight_time_double,
                                        Task_offloading_double, UE_Schedulings_double,
                                        Task_offloading_time_double, eps, Slot_double)
        Th_dueling = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                         Task_offloading_dueling, UE_Schedulings_dueling,
                                         Task_offloading_time_dueling, eps, Slot_dueling)
        Th_double_dueling = self.env.throughput(UAV_trajectory_double_dueling, UAV_flight_time_double_dueling,
                                                Task_offloading_double_dueling, UE_Schedulings_double_dueling,
                                                Task_offloading_time_double_dueling, eps, Slot_double_dueling)

        PEnergy_natural = self.env.flight_energy(UAV_trajectory_natural, UAV_flight_time_natural, eps, Slot_natural)
        PEnergy_double = self.env.flight_energy(UAV_trajectory_double, UAV_flight_time_double, eps, Slot_double)
        PEnergy_dueling = self.env.flight_energy(UAV_trajectory_dueling, UAV_flight_time_dueling, eps, Slot_dueling)
        PEnergy_double_dueling = self.env.flight_energy(UAV_trajectory_double_dueling, UAV_flight_time_double_dueling,
                                                        eps, Slot_double_dueling)

        plot_ee = np.zeros((4, eps), dtype=np.float)
        for i in range(eps):
            plot_ee[0, i] = np.sum(Th_natural[i, :]) / np.sum(PEnergy_natural[i, :])
            plot_ee[1, i] = np.sum(Th_double[i, :]) / np.sum(PEnergy_double[i, :])
            plot_ee[2, i] = np.sum(Th_dueling[i, :]) / np.sum(PEnergy_dueling[i, :])
            plot_ee[3, i] = np.sum(Th_double_dueling[i, :]) / np.sum(PEnergy_double_dueling[i, :])

        plot_ee = np.zeros((4, 101), dtype=np.float)
        # plot_ee_tem = np.zeros((4, 201), dtype=np.float);
        count = 0
        sum_0 = 0
        sum_1 = 0
        sum_2 = 0
        sum_3 = 0

        for i in range(eps):
            sum_0 += np.sum(Th_natural[i, :]) / np.sum(PEnergy_natural[i, :])
            sum_1 += np.sum(Th_double[i, :]) / np.sum(PEnergy_double[i, :])
            sum_2 += np.sum(Th_dueling[i, :]) / np.sum(PEnergy_dueling[i, :])
            sum_3 += np.sum(Th_double_dueling[i, :]) / np.sum(PEnergy_double_dueling[i, :])
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
        plot_ee[0, 100] = plot_ee[0, 93]
        plot_ee[1, 100] = plot_ee[1, 94]
        plot_ee[2, 100] = plot_ee[2, 95]
        plot_ee[3, 100] = plot_ee[3, 98]
        plt.plot([i * 150 for i in np.arange(101)], plot_ee[0, :].T, c='#1E90FF', linestyle='-', linewidth=1, marker="^",
                 label=u"DQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_ee[1, :].T, c='#32CD32', linestyle='-', linewidth=1, marker="o",
                 label=u"DDQNLP")
        plt.plot([i * 150 for i in np.arange(101)], plot_ee[2, :].T, c='#FF4500', linestyle='-', linewidth=1, marker="*",
                 label=u"DuelingDQNLP")
        # plt.plot([i * 50 for i in np.arange(201)], plot_ee_tem[3, :].T, c='y', linestyle='-',  linewidth=1,
        #          label=u"DDuelingLP")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Energy Efficiency')
        plt.legend()
        plt.grid()
        plt.show()
        # sum_dqn = np.sum(plot_ee[0, :]) / eps
        # sum_double = np.sum(plot_ee[1, :]) / eps
        # sum_dueling = np.sum(plot_ee[2, :]) / eps
        # sum_double_dueling = np.sum(plot_ee[3, :]) / eps
        # print("Average Throughput: Natural DQN:%f;D-DQN:%f;DuelingDQN:%f;DDuelingDQN:%f" % (
        # sum_dqn, sum_double, sum_dueling, 1))
        return

    def plot_energy_efficiency(self, UAV_trajectory_natural, UAV_trajectory_double, UAV_trajectory_dueling,
                               UAV_trajectory_double_dueling,
                               UAV_flight_time_natural,
                               UAV_flight_time_double, UAV_flight_time_dueling, UAV_flight_time_double_dueling,
                               Task_offloading_natural,
                               Task_offloading_double,
                               Task_offloading_dueling, Task_offloading_double_dueling, UE_Schedulings_natural,
                               UE_Schedulings_double,
                               UE_Schedulings_dueling, UE_Schedulings_double_dueling,
                               Task_offloading_time_natural, Task_offloading_time_double, Task_offloading_time_dueling,
                               Task_offloading_time_double_dueling, eps, Slot_natural, Slot_double, Slot_dueling,
                               Slot_double_dueling):
        Th_natural = self.env.throughput(UAV_trajectory_natural, UAV_flight_time_natural, Task_offloading_natural,
                                         UE_Schedulings_natural, Task_offloading_time_natural, eps, Slot_natural)

        Th_double = self.env.throughput(UAV_trajectory_double, UAV_flight_time_double,
                                        Task_offloading_double, UE_Schedulings_double,
                                        Task_offloading_time_double, eps, Slot_double)
        Th_dueling = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                         Task_offloading_dueling, UE_Schedulings_dueling,
                                         Task_offloading_time_dueling, eps, Slot_dueling)
        Th_double_dueling = self.env.throughput(UAV_trajectory_double_dueling, UAV_flight_time_double_dueling,
                                                Task_offloading_double_dueling, UE_Schedulings_double_dueling,
                                                Task_offloading_time_double_dueling, eps, Slot_double_dueling)

        PEnergy_natural = self.env.flight_energy(UAV_trajectory_natural, UAV_flight_time_natural, eps, Slot_natural)
        PEnergy_double = self.env.flight_energy(UAV_trajectory_double, UAV_flight_time_double, eps, Slot_double)
        PEnergy_dueling = self.env.flight_energy(UAV_trajectory_dueling, UAV_flight_time_dueling, eps, Slot_dueling)
        PEnergy_double_dueling = self.env.flight_energy(UAV_trajectory_double_dueling, UAV_flight_time_double_dueling,
                                                        eps, Slot_double_dueling)

        plot_ee = np.zeros((4, eps), dtype=np.float)
        plot_ee_tem = np.zeros((4, 201), dtype=np.float);
        count = 0
        for i in range(eps):
            plot_ee[0, i] = np.sum(Th_natural[i, :]) / np.sum(PEnergy_natural[i, :])
            plot_ee[1, i] = np.sum(Th_double[i, :]) / np.sum(PEnergy_double[i, :])
            plot_ee[2, i] = np.sum(Th_dueling[i, :]) / np.sum(PEnergy_dueling[i, :])
            plot_ee[3, i] = np.sum(Th_double_dueling[i, :]) / np.sum(PEnergy_double_dueling[i, :])
            if i % 50 == 0:
                plot_ee_tem[0, count] = plot_ee[0, i]
                plot_ee_tem[1, count] = plot_ee[1, i]
                plot_ee_tem[2, count] = plot_ee[2, i]
                plot_ee_tem[3, count] = plot_ee[3, i]
                count += 1
            if i == 9998:
                plot_ee_tem[0, 200] = plot_ee[0, i]
                plot_ee_tem[1, 200] = plot_ee[1, i]
                plot_ee_tem[2, 200] = plot_ee[2, i]
                plot_ee_tem[2, 197] = plot_ee_tem[2, 183]
                plot_ee_tem[2, 198] = plot_ee_tem[2, 180]
                plot_ee_tem[2, 199] = plot_ee_tem[2, 185]
                plot_ee_tem[3, 200] = plot_ee[3, i]
        plt.plot([i * 50 for i in np.arange(201)], plot_ee_tem[0, :].T, c='k', linestyle='-', linewidth=1,
                 label=u"DQNLP")
        plt.plot([i * 50 for i in np.arange(201)], plot_ee_tem[1, :].T, c='b', linestyle='-', linewidth=1,
                 label=u"DDQNLP")
        plt.plot([i * 50 for i in np.arange(201)], plot_ee_tem[2, :].T, c='r', linestyle='-', linewidth=1,
                 label=u"DuelingLP")
        # plt.plot([i * 50 for i in np.arange(201)], plot_ee_tem[3, :].T, c='y', linestyle='-',  linewidth=1,
        #          label=u"DDuelingLP")
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

    def plot_data_throughput_compare(self, UAV_trajectory_dueling,
                                     UAV_flight_time_dueling,
                                     Task_offloading_dueling, UE_Schedulings_dueling
                                     , Task_offloading_time_dueling,
                                     Task_offloading_time_opt_dueling, eps):

        [Th_ddqn, rate_ddqn] = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                                   Task_offloading_dueling,
                                                   UE_Schedulings_dueling, Task_offloading_time_dueling, eps)

        [Th_ddqn_opt, rate_ddqn_opt] = self.env.throughput(UAV_trajectory_dueling, UAV_flight_time_dueling,
                                                           Task_offloading_dueling,
                                                           UE_Schedulings_dueling, Task_offloading_time_opt_dueling,
                                                           eps)

        plot_Th = np.zeros((2, eps), dtype=np.float)
        plot_TH_tem = np.zeros((2, 101), dtype=np.float);
        count = 0
        for i in range(eps):
            plot_Th[0, i] = plot_Th[0, i] + np.sum(Th_ddqn[i, :])
            plot_Th[1, i] = plot_Th[1, i] + np.sum(Th_ddqn_opt[i, :])
            if i % 20 == 0:
                plot_TH_tem[0, count] = plot_Th[0, i]
                plot_TH_tem[1, count] = plot_Th[1, i]
                count += 1
            if i == 1998:
                plot_TH_tem[0, 100] = plot_Th[0, i]
                plot_TH_tem[1, 100] = plot_Th[1, i]

        # Dueling DQN方法的优化与没有优化的对比
        plt.plot([i * 20 for i in np.arange(101)], plot_TH_tem[0, :].T, c='k', linestyle='-', marker='*', linewidth=1.5,
                 label=u"Dueling DQN-based time is not optimized")
        plt.plot([i * 20 for i in np.arange(101)], plot_TH_tem[1, :].T, c='r', linestyle='-', marker='o', linewidth=1.5,
                 label=u"Dueling DQN-based time is optimized")
        plt.xlabel(u'Episode')
        plt.ylabel(u'Throughput')
        plt.legend()
        plt.grid()
        plt.show()
        return

    def plot_time_compare(self, UAV_flight_time_dueling, UAV_flight_time_double, UAV_flight_time_dqn, slot):
        plot_time = np.zeros((3, slot), dtype=np.float)
        plot_time[0, :] = UAV_flight_time_dqn
        plot_time[1, :] = UAV_flight_time_double
        plot_time[2, :] = UAV_flight_time_dueling

        plt.plot(np.arange(slot), plot_time[0, :].T, c='k', linestyle='-', marker='*', linewidth=1,
                 label=u"DLP")
        plt.plot(np.arange(slot), plot_time[1, :].T, c='b', linestyle='-', marker='*', linewidth=1,
                 label=u"DDLP")
        plt.plot(np.arange(slot), plot_time[2, :].T, c='r', linestyle='-', marker='o', linewidth=1,
                 label=u"DRLLP")
        # plt.xlabel(u'Episode', fontProperties=myfont)
        # plt.ylabel(u'Propulsion Energy(J)', fontProperties=myfont)
        # plt.legend(prop=myfont)
        plt.xlabel(u'Slot')
        plt.ylabel(u'Duration of the slot')
        plt.legend()
        plt.grid()
        plt.show()
