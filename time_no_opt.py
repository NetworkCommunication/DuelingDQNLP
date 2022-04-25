import numpy as np

from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare
import math as mt

#
env = UAV_MEC()
res = Res_plot_compare(env)

if __name__ == '__main__':
    # Slot_natural = np.load("Slot_natural.npy")
    # Slot_double = np.load("Slot_double.npy")
    # Slot_dueling = np.load("Slot_dueling.npy")
    # Slot_double_dueling = np.load("Slot_double_dueling.npy")
    # # print(Slot_natural)
    # # print(Slot_double)
    # # print(Slot_dueling)
    # # print(Slot_double_dueling)
    # UAV_trajectory_natural = np.load("UAV_trajectory_natural.npy")
    # UAV_trajectory_double = np.load("UAV_trajectory_double.npy")
    # UAV_trajectory_dueling = np.load("UAV_trajectory_dueling.npy")
    # UAV_trajectory_double_dueling = np.load("UAV_trajectory_double_dueling.npy")
    # # print(UAV_trajectory_natural)
    # # print(UAV_trajectory_double)
    # # print(UAV_trajectory_dueling)
    # for i in range(len(UAV_trajectory_double_dueling)):
    #     print(UAV_trajectory_double_dueling[88,:,2])
    # # print(np.sum(Slot_natural[0,:]))
    # # print(np.sum(Slot_double[0,:]))
    # # print(np.sum(Slot_dueling[0,:]))
    # # print(np.sum(Slot_double_dueling[0,:]))
    # for j in range(np.int32(1 / 0.1), np.int32(3 / 0.1) + 1, 5):
    #     print(j)

    # min = 100000
    # index = -1
    # for gt in range(env.GTs):
    #     h = env.l_o_v + 3 * env.h_s
    #     x = 4 * env.x_s + 0.5 * env.x_s
    #     y = 5 * env.y_s + 0.5 * env.y_s
    #     d = np.sqrt(mt.pow(h, 2) + mt.pow(x - env.w_k[gt, 0], 2) + mt.pow(y - env.w_k[gt, 1], 2))
    #     print(d)
    #     if min > d:
    #         min=d
    #         index = gt
    #
    # print(index)

    Task_offloading_dueling = np.load("DDD/Task_offloading_dueling.npy")
    # Task_offloading_dueling = Task_offloading_dueling[:10000, :, :]
    Slot_dueling = np.load("DDD/Slot_dueling.npy")
    # Slot_dueling = Slot_dueling[:10000, :]
    UAV_flight_time_dueling = np.load("DDD/UAV_flight_time_dueling.npy")
    # UAV_flight_time_dueling = UAV_flight_time_dueling[:10000, :]
    Task_offloading_time_no_opt_dueling = np.zeros((20000, env.N_slot, env.TKs), dtype=np.float32)  # 任务时间分配
    for i in range(env.eps - 1):
        for slot in range(env.N_slot):
            if slot < Slot_dueling[0, i]:
                offloading_action = Task_offloading_dueling[i, slot, :]
                count_nonzero = np.count_nonzero(offloading_action)
                t_n_t_c_tmp = np.zeros((1, env.TKs), dtype=np.float32)
                if count_nonzero != 0:
                    tmp = (UAV_flight_time_dueling[i, slot] / (count_nonzero))
                    t_n_t_c_tmp[0][np.nonzero(offloading_action)] = tmp
                t_n_t_c = t_n_t_c_tmp[0]
                Task_offloading_time_no_opt_dueling[i, slot, :] = t_n_t_c
    np.save(file="Task_offloading_time_no_opt_dueling.npy", arr=Task_offloading_time_no_opt_dueling)
    # UAV_flight_time_dueling = np.load("DDD/Task_offloading_time_no_opt_dueling.npy")
    # print(Task_offloading_time_no_opt_dueling)

