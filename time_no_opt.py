import numpy as np

from UAV_MEC_env import UAV_MEC
from Res_plot_compare import Res_plot_compare

env = UAV_MEC()
res = Res_plot_compare(env)

if __name__ == '__main__':

    Task_offloading_dueling = np.load("DDD/Task_offloading_dueling.npy")
    Slot_dueling = np.load("DDD/Slot_dueling.npy")
    UAV_flight_time_dueling = np.load("DDD/UAV_flight_time_dueling.npy")
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

