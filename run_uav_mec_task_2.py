
from UAV_MEC_env import UAV_MEC
import numpy as np
import tensorflow as tf
from scipy.optimize import linprog  # 导入 scipy
from RL_brain_dueling import DoubleDQN
import math as mt

env = UAV_MEC()
res = Res_plot_compare(env)
MEMORY_SIZE = 3200
Episodes = env.eps

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, dueling=0, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, dueling=1, sess=sess, output_graph=True)

with tf.variable_scope('Dueling_DQN'):
    dueling_DQN = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, dueling=2, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())

# record the results

# 优化的是无人机的轨迹以及任务的卸载
UAV_trajectory_natural = np.zeros((Episodes, env.N_slot, 3), dtype=np.float32)
Task_offloading_natural = np.zeros((Episodes, env.N_slot, env.TKs), dtype=np.float32)
UAV_flight_time_natural = np.ones((Episodes, env.N_slot), dtype=np.float32)
UE_Schedulings_natural = np.zeros((Episodes, env.N_slot), dtype=np.float32)  # 用户调度
Task_offloading_time_natural = np.ones((Episodes, env.N_slot, env.TKs), dtype=np.float32)
Convergence_natural = np.ones((Episodes, env.N_slot), dtype=np.float32)
Slot_natural = np.zeros((1, Episodes), dtype=np.int)

UAV_trajectory_double = np.zeros((Episodes, env.N_slot, 3), dtype=np.float32)
Task_offloading_double = np.ones((Episodes, env.N_slot, env.TKs), dtype=np.float32)
UAV_flight_time_double = np.ones((Episodes, env.N_slot), dtype=np.float32)
UE_Schedulings_double = np.zeros((Episodes, env.N_slot), dtype=np.float32)  # 用户调度
Task_offloading_time_double = np.ones((Episodes, env.N_slot, env.TKs), dtype=np.float32)  # 任务时间分配
Convergence_double = np.ones((Episodes, env.N_slot), dtype=np.float32)
Slot_double = np.zeros((1, Episodes), dtype=np.int)

UAV_trajectory_dueling = np.zeros((Episodes, env.N_slot, 3), dtype=np.float32)
Task_offloading_dueling = np.ones((Episodes, env.N_slot, env.TKs), dtype=np.float32)
UAV_flight_time_dueling = np.ones((Episodes, env.N_slot), dtype=np.float32)
UE_Schedulings_dueling = np.zeros((Episodes, env.N_slot), dtype=np.float32)  # 用户调度
Task_offloading_time_dueling = np.ones((Episodes, env.N_slot, env.TKs), dtype=np.float32)  # 任务时间分配
Convergence_dueling = np.ones((Episodes, env.N_slot), dtype=np.float32)
Slot_dueling = np.zeros((1, Episodes), dtype=np.int)


def train(RL):
    total = 0
    for ep in range(Episodes):
        observation = env.reset()
        slot = 0
        while (env.finish == False):
            min = 100000
            index_wk = -1
            for gt in range(env.GTs):
                h = env.l_o_v + observation[2] * env.h_s
                x = observation[0] * env.x_s + 0.5 * env.x_s
                y = observation[1] * env.y_s + 0.5 * env.y_s
                d = np.sqrt(mt.pow(h, 2) + mt.pow(x - env.w_k[gt, 0], 2) + mt.pow(y - env.w_k[gt, 1], 2))
                # print(d)
                if min > d:
                    min = d
                    index_wk = gt
            action_index = RL.choose_action(observation)
            action = env.find_action(action_index)
            c = np.zeros((env.TKs), dtype=np.float32)
            A_eq = np.array([[0, 0, 0, 0, 0, 0]])  # 等式约束参数矩阵
            B_eq = np.array([action[-1] / 10])  # 等式约束参数向量
            b = [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)]
            h = observation[2]
            x = observation[0]
            y = observation[1]
            for j in range(env.GTs):
                if j == index_wk:
                    r_g = env.link_rate_single(h, x, y, env.w_k[j, :])
                    for k in range(env.TKs):
                        a_kn = action[k + 3]  # 需要加一个数
                        if a_kn != 0:
                            a_0 = -(a_kn * env.f_u * env.u_k[j, k, 0] * r_g) / (
                                    r_g * env.u_k[j, k, 1] + env.f_u * env.u_k[j, k, 0])
                            c[k] = a_0
                            A_eq[0][k] = 1
                            b[k] = (0, action[-1] / 10)
            res = linprog(c, A_ub=None, b_ub=None, A_eq=A_eq, b_eq=B_eq,
                          bounds=(b[0], b[1], b[2], b[3], b[4], b[5]))
            t_n_t_c = res['x'][:]
            observation_, reward = env.step(action, t_n_t_c, slot, index_wk)
            RL.store_transition(observation, action, reward, observation_)
            if (RL.dueling == 1):
                UAV_trajectory_double[ep, slot, :] = observation_[:]  # 里面存的是 [self.l_n[0], self.l_n[1], self.h_n]
                UE_Schedulings_double[ep, slot] = index_wk
                Task_offloading_double[ep, slot, :] = action[-env.TKs - 1:-1]
                UAV_flight_time_double[ep, slot] = action[-1] / 10
                Task_offloading_time_double[ep, slot, :] = t_n_t_c
                Convergence_double[ep, slot] = reward
                Slot_double[0, ep] = Slot_double[0, ep] + 1
            if (RL.dueling == 0):
                UAV_trajectory_natural[ep, slot, :] = observation_[:]
                UE_Schedulings_natural[ep, slot] = index_wk
                Task_offloading_natural[ep, slot, :] = action[-env.TKs - 1:-1]
                UAV_flight_time_natural[ep, slot] = action[-1] / 10
                Task_offloading_time_natural[ep, slot, :] = t_n_t_c
                Convergence_natural[ep, slot] = reward
                Slot_natural[0, ep] = Slot_natural[0, ep] + 1
            if (RL.dueling == 2):
                UAV_trajectory_dueling[ep, slot, :] = observation_[:]
                UE_Schedulings_dueling[ep, slot] = index_wk
                Task_offloading_dueling[ep, slot, :] = action[-env.TKs - 1:-1]
                UAV_flight_time_dueling[ep, slot] = action[-1] / 10
                Task_offloading_time_dueling[ep, slot, :] = t_n_t_c
                Convergence_dueling[ep, slot] = reward
                Slot_dueling[0, ep] = Slot_dueling[0, ep] + 1
            if total + slot >= MEMORY_SIZE:
                RL.learn()
            observation = observation_  # 进入下一个状态
            slot = slot + 1
            total = total + 1
        print("Finish episode %d" % ep)
        if (RL.dueling == 1):
            UAV_trajectory_double[ep, :] = env.UAV_FLY(UAV_trajectory_double[ep, :], Slot_double[0, ep])
        if (RL.dueling == 0):
            UAV_trajectory_natural[ep, :] = env.UAV_FLY(UAV_trajectory_natural[ep, :], Slot_natural[0, ep])
        if (RL.dueling == 2):
            UAV_trajectory_dueling[ep, :] = env.UAV_FLY(UAV_trajectory_dueling[ep, :], Slot_dueling[0, ep])
    return RL.q


q_natural = train(natural_DQN)
q_double = train(double_DQN)
q_dueling = train(dueling_DQN)
# Double DQN
np.save(file="UAV_trajectory_double.npy", arr=UAV_trajectory_double)
np.save(file="UE_Schedulings_double.npy", arr=UE_Schedulings_double)
np.save(file="Task_offloading_double.npy", arr=Task_offloading_double)
np.save(file="UAV_flight_time_double.npy", arr=UAV_flight_time_double)
np.save(file="Task_offloading_time_double.npy", arr=Task_offloading_time_double)
np.save(file="Convergence_double.npy", arr=Convergence_double)
np.save(file="Slot_double.npy", arr=Slot_double)

# Dueling DQN
np.save(file="UAV_trajectory_dueling.npy", arr=UAV_trajectory_dueling)
np.save(file="UE_Schedulings_dueling.npy", arr=UE_Schedulings_dueling)
np.save(file="Task_offloading_dueling.npy", arr=Task_offloading_dueling)
np.save(file="UAV_flight_time_dueling.npy", arr=UAV_flight_time_dueling)
np.save(file="Task_offloading_time_dueling.npy", arr=Task_offloading_time_dueling)
np.save(file="Convergence_dueling.npy", arr=Convergence_dueling)
np.save(file="Slot_dueling.npy", arr=Slot_dueling)

# DQN
np.save(file="UAV_trajectory_natural.npy", arr=UAV_trajectory_natural)
np.save(file="UE_Schedulings_natural.npy", arr=UE_Schedulings_natural)
np.save(file="Task_offloading_natural.npy", arr=Task_offloading_natural)
np.save(file="UAV_flight_time_natural.npy", arr=UAV_flight_time_natural)
np.save(file="Task_offloading_time_natural.npy", arr=Task_offloading_time_natural)
np.save(file="Convergence_natural.npy", arr=Convergence_natural)
np.save(file="Slot_natural.npy", arr=Slot_natural)
EPS = env.eps - 1
