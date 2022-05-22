import numpy as np
import random as rd
import time
import math as mt
import sys
import copy

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib.pyplot as plt

UNIT = 1  # pixels
IOT_H = 1000  # grid height
IOT_W = 1000  # grid width
Max_Hight = 200  # maximum level of height
Min_Hight = 100  # minimum level of height

# weight variables for the reward function
beta = 20

D_k = 1000  # 处理的数量总数
F_k = 2000  # CPU周期总数

N_0 = mt.pow(10, ((-169 / 3) / 10))
a = 9.61
b = 0.16
eta_los = 1
eta_nlos = 20
A = eta_los - eta_nlos
C = 20 * np.log10(
    4 * np.pi * 9 / 3) + eta_nlos
B = 2000
Power = 5 * mt.pow(10, 5)
t_min = 1
t_max = 3


class UAV_MEC(object):
    def __init__(self):
        super(UAV_MEC, self).__init__()
        self.N_slot = 400  # number of time slots in one episode
        self.x_s = 10
        self.y_s = 10  # 网格化地图后每一格的长度
        self.h_s = 2  # 每一段的高度
        self.GTs = 6
        self.TKs = 6
        self.l_o_v = 50 * self.h_s  # initial vertical location
        self.l_f_v = 50 * self.h_s  # final vertical location
        self.l_o_h = [0, 0]  # initial horizontal location 无人机起始水平位置
        self.l_f_h = [100, 100]
        self.eps = 15000  # number of episode
        self.energy_max = 40000

        self.UAV_trajectory_tsp = np.zeros((self.N_slot, 3), dtype=np.float)  # 初始化无人机轨迹

        self.f_u = 100  # 无人机计算速率
        self.f_g = 5  # 用户计算速率
        # 最大距离
        self.D_max = np.sqrt(mt.pow(self.l_o_h[0] - self.l_f_h[0], 2) + mt.pow(
            self.l_o_h[1] - self.l_f_h[1], 2))

        # up down left right static
        # self.action_space_uav_horizontal = ['n', 's', 'e','w','h']
        self.action_space_uav_horizontal = ['ul', 'ur', 'us', 'dl', 'dr', 'ds', 'sl', 'sr', 'ss']
        # ascend, descend, slf
        self.action_space_uav_vertical = ['a', 'd', 's']
        # offloading, local exection
        self.action_space_task_offloading = np.zeros((self.GTs, 2), dtype=np.int)
        # overall_action_space
        self.n_actions = len(self.action_space_uav_horizontal) * len(self.action_space_uav_vertical) * mt.pow(2,
                                                                                                              self.TKs) * 5
        self.n_features = 3  # horizontal:x, y, vertical trajectory of the UAV

        self.actions = np.zeros((np.int(self.n_actions), 1 + 2 + self.TKs + 1), dtype=np.int)
        index = 0
        # 这段代码初始化了动作的空间
        for h in range(len(self.action_space_uav_horizontal)):  # 水平位置
            for v in range(len(self.action_space_uav_vertical)):  # 垂直位置
                # for s in range(self.GTs):
                LL = self.brgd(
                    self.TKs)  # list all the possible combination of 0-1 offloading options among the GTs 64种情况
                for l in range(len(LL)):  # 任务的卸载
                    o_string = LL[l]
                    of = []
                    for ch in range(len(o_string)):
                        if o_string[ch] == '0':
                            of.append(0)
                        else:
                            of.append(1)
                    for j in range(np.int32(t_min / 0.1), np.int32((t_max) / 0.1) + 1, 5):
                        self.actions[index, :] = [index, h, v] + of[:] + [j]
                        index = index + 1
        self._build_uav_mec()

    def _build_uav_mec(self):
        a = [783, 364, 612, 318, 198, 521]
        b = [614, 415, 210, 827, 100, 885]
        d_k = np.load("DK.npy")
        f_k = np.load("FK.npy")
        self.w_k = np.zeros((self.GTs, 2), dtype=np.float32)
        self.u_k = np.zeros((self.GTs, self.TKs, 2), dtype=np.float32)

        for g in range(self.GTs):
            self.w_k[g, 0] = a[g]
            self.w_k[g, 1] = b[g]
        for g in range(self.GTs):
            for k in range(self.TKs):
                self.u_k[g, k, 0] = d_k[g, k]
                self.u_k[g, k, 1] = f_k[g, k]
        return

    # 重新开始的无人机的位置
    def reset(self):
        # reset the UAV
        self.d_s = np.zeros((self.N_slot, self.GTs), dtype=np.float)  # data processed
        self.energy = np.zeros((1, self.N_slot), dtype=np.float)  # propulsion  energy of the UAV
        self.finish = False
        self.slot = 0
        self.h_n = 0
        self.l_n = [0, 0]
        return np.array([self.l_n[0], self.l_n[1], self.h_n])

    # 传输速率
    def link_rate(self, gt):
        h = self.l_o_v + self.h_n * self.h_s
        x = self.l_n[0] * self.x_s + 0.5 * self.x_s
        y = self.l_n[1] * self.y_s + 0.5 * self.y_s
        d = np.sqrt(mt.pow(h, 2) + mt.pow(x - self.w_k[gt, 0], 2) + mt.pow(y - self.w_k[gt, 1], 2))

        if (np.sqrt(mt.pow(x - self.w_k[gt, 0], 2) + mt.pow(y - self.w_k[gt, 1], 2)) > 0):
            ratio = h / np.sqrt(mt.pow(x - self.w_k[gt, 0], 2) + mt.pow(y - self.w_k[gt, 1], 2))
        else:
            ratio = np.Inf

        p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
        p_los = 1 / p_los
        L_km = 20 * np.log10(d) + A * p_los + C
        r = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
        return r

    # 传输速率只是参数不同
    def link_rate_single(self, hh, xx, yy, w_k):
        h = self.l_o_v + hh * self.h_s
        x = xx * self.x_s + 0.5 * self.x_s
        y = yy * self.y_s + 0.5 * self.y_s
        d = np.sqrt(mt.pow(h, 2) + mt.pow(x - w_k[0], 2) + mt.pow(y - w_k[1], 2))
        if (np.sqrt(mt.pow(x - w_k[0], 2) + mt.pow(y - w_k[1], 2)) > 0):
            ratio = h / np.sqrt(mt.pow(x - w_k[0], 2) + mt.pow(y - w_k[1], 2))
        else:
            ratio = np.Inf
        p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
        p_los = 1 / p_los
        L_km = 20 * np.log10(d) + A * p_los + C
        r = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
        return r

    def step(self, action, t_n_t, slot,index_wk):
        self.finish = True
        h = action[1]
        v = action[2]
        t_n = action[1 + 2 + 6]

        pre_l_n = self.l_n
        pre_h_n = self.h_n

        # update height of the UAV
        self.OtPoI = 0
        if v == 0:  # ascending
            self.h_n = self.h_n + 1
            if self.h_n > 50:
                self.h_n = self.h_n - 1
                self.OtPoI = 1
        elif v == 1:  # descending
            self.h_n = self.h_n - 1
            if self.h_n < 0:
                self.h_n = self.h_n + 1
                self.OtPoI = 1
        elif v == 2:  # SLF
            self.h_n = self.h_n

        # fly action=5
        # if h == 0:  # north
        #     self.l_n[1] = self.l_n[1] + 1
        #     if self.l_n[1] > 100:  # if out of PoI
        #         self.l_n[1] = self.l_n[1] - 1
        #         self.OtPoI = 1
        # elif h == 1:  # south
        #     self.l_n[1] = self.l_n[1] - 1
        #     if self.l_n[1] < 0:  # if out of PoI
        #         self.l_n[1] = self.l_n[1] + 1
        #         self.OtPoI = 1  # 自己加的
        # elif h == 2:  # east
        #     self.l_n[0] = self.l_n[0] + 1
        #     if self.l_n[0] > 100:  # if out of PoI
        #         self.l_n[0] = self.l_n[0] - 1
        #         self.OtPoI = 1
        # elif h == 3:  # west
        #     self.l_n[0] = self.l_n[0] - 1
        #     if self.l_n[0] < 0:  # if out of PoI
        #         self.l_n[0] = self.l_n[0] + 1
        #         self.OtPoI = 1
        # elif h == 4:  # hover
        #     self.l_n[0] = self.l_n[0]
        #     self.l_n[1] = self.l_n[1]

        # fly action=9
        if h == 0:  # up left
            self.l_n[1] = self.l_n[1] + 1
            self.l_n[0] = self.l_n[0] - 1
            if self.l_n[1] > 100:  # if out of PoI
                self.l_n[1] = self.l_n[1] - 1
                self.OtPoI = 1
            if self.l_n[0] < 0:  # if out of PoI
                self.l_n[0] = self.l_n[0] + 1
                self.OtPoI = 1
        elif h == 1:  # up right
            self.l_n[1] = self.l_n[1] + 1
            self.l_n[0] = self.l_n[0] + 1
            if self.l_n[1] > 100:  # if out of PoI
                self.l_n[1] = self.l_n[1] - 1
                self.OtPoI = 1
            if self.l_n[0] > 100:  # if out of PoI
                self.l_n[0] = self.l_n[0] - 1
                self.OtPoI = 1
        elif h == 2:  # up static
            self.l_n[1] = self.l_n[1] + 1
            self.l_n[0] = self.l_n[0]
            if self.l_n[0] > 100:  # if out of PoI
                self.l_n[0] = self.l_n[0] - 1
                self.OtPoI = 1
        elif h == 3:  # down left
            self.l_n[1] = self.l_n[1] - 1
            self.l_n[0] = self.l_n[0] - 1
            if self.l_n[1] < 0:  # if out of PoI
                self.l_n[1] = self.l_n[1] + 1
                self.OtPoI = 1
            if self.l_n[0] < 0:  # if out of PoI
                self.l_n[0] = self.l_n[0] + 1
                self.OtPoI = 1
        elif h == 4:  # down right
            self.l_n[1] = self.l_n[1] - 1
            self.l_n[0] = self.l_n[0] + 1
            if self.l_n[1] < 0:  # if out of PoI
                self.l_n[1] = self.l_n[1] + 1
                self.OtPoI = 1
            if self.l_n[0] > 100:  # if out of PoI
                self.l_n[0] = self.l_n[0] - 1
                self.OtPoI = 1
        elif h == 5:  # down static
            self.l_n[1] = self.l_n[1] - 1
            self.l_n[0] = self.l_n[0]
            if self.l_n[1] < 0:  # if out of PoI
                self.l_n[1] = self.l_n[1] + 1
                self.OtPoI = 1
        elif h == 6:  # static left
            self.l_n[1] = self.l_n[1]
            self.l_n[0] = self.l_n[0] - 1
            if self.l_n[0] < 0:  # if out of PoI
                self.l_n[0] = self.l_n[0] + 1
                self.OtPoI = 1
        elif h == 7:  # static right
            self.l_n[1] = self.l_n[1]
            self.l_n[0] = self.l_n[0] + 1
            if self.l_n[0] > 100:  # if out of PoI
                self.l_n[0] = self.l_n[0] - 1
                self.OtPoI = 1
        elif h == 8:  # static static
            self.l_n[0] = self.l_n[0]
            self.l_n[1] = self.l_n[1]

        a_kn = np.zeros((1, self.TKs), dtype=np.int32)  # 卸载决策
        d_s = np.zeros((1, self.TKs),
                       dtype=np.float32)
        self.energy[0, slot] = self.flight_energy_slot(pre_l_n, self.l_n, pre_h_n, self.h_n, t_n)
        cumulative_energy = sum(self.energy[0, :])
        if (cumulative_energy < self.energy_max):
            self.finish = False
        for t in range(self.TKs):
            a_kn[0, t] = action[1 + 2 + t]
        for g in range(self.GTs):
            if g == index_wk:
                for t in range(self.TKs):
                    d_s[0, t] = t_n_t[t] * a_kn[0, t] * ((self.f_u * self.u_k[g, t, 0] * self.link_rate(g)) / (
                            self.link_rate(g) * self.u_k[g, t, 1] + self.f_u * self.u_k[g, t, 0])) + (
                                        1 - a_kn[0, t]) * self.f_g * (t_n / 10)
        reward = np.sum(d_s[0, :]) / self.energy[0, slot]
        if self.OtPoI == 1:
            reward = reward - 0.1  # give an additional penality if out of PoI: P=0.3
        _state = np.array([self.l_n[0], self.l_n[1], self.h_n])
        return _state, reward

    def find_action(self, index):
        return self.actions[index, :]

    # 卸载情况，['000000', '000001', '000011', '000010', '000110', '000111'...]
    def brgd(self, n):
        if n == 1:
            return ["0", "1"]
        L1 = self.brgd(n - 1)
        L2 = copy.deepcopy(L1)
        L2.reverse()
        L1 = ["0" + l for l in L1]
        L2 = ["1" + l for l in L2]
        L = L1 + L2
        return L

    def flight_energy(self, UAV_trajectory, UAV_flight_time, EP, slot):
        d_o = 0.6  # fuselage equivalent flat plate area;
        rho = 1.225  # air density in kg/m3;
        s = 0.05  # rotor solidity;
        G = 0.503  # Rotor disc area in m2;
        U_tip = 120  # tip seep of the rotor blade(m/s);
        v_o = 4.3  # mean rotor induced velocity in hover;
        omega = 300  # blade angular velocity in radians/second;
        R = 0.4  # rotor radius in meter;
        delta = 0.012  # profile drage coefficient;
        k = 0.1  # incremental correction factor to induced power;
        W = 20  # aircraft weight in newton;
        P0 = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
        P1 = (1 + k) * (pow(W, (3 / 2)) / np.sqrt(2 * rho * G))
        Energy_uav = np.zeros((EP, self.N_slot), dtype=np.float32)
        P2 = 11.46
        count = 0
        for ep in range(self.eps - EP, self.eps):
            horizontal = UAV_trajectory[ep, :, [0, 1]]
            vertical = UAV_trajectory[ep, :, -1]
            t_n = UAV_flight_time[ep, :]

            for i in range(slot[0, ep]):
                if (i == 0):
                    d = np.sqrt((horizontal[0, i] - self.l_o_h[0]) ** 2 + (
                            horizontal[1, i] - self.l_o_h[1]) ** 2)
                    h = np.abs(vertical[i] - vertical[0])
                else:
                    d = np.sqrt(
                        (horizontal[0, i] - horizontal[
                            0, i - 1]) ** 2 + (
                                horizontal[1, i] - horizontal[
                            1, i - 1]) ** 2)
                    h = np.abs(vertical[i] - vertical[i - 1])

                v_h = d / t_n[i]
                v_v = h / t_n[i]
                Energy_uav[count, i] = t_n[i] * P0 * (1 + 3 * np.power(v_h, 2) / np.power(U_tip, 2)) + t_n[i] * (
                        1 / 2) * d_o * rho * s * G * np.power(v_h, 3) + \
                                       t_n[i] * P1 * np.sqrt(
                    np.sqrt(1 + np.power(v_h, 4) / (4 * np.power(v_o, 4))) - np.power(v_h, 2) / (
                            2 * np.power(v_o, 2))) + P2 * v_v * t_n[i]
            count = count + 1
        return Energy_uav

    # 一个时间间隙中的能耗
    def flight_energy_slot(self, pre_l_n, l_n, pre_h, h, t_n):
        d_o = 0.6  # fuselage equivalent flat plate area;
        rho = 1.225  # air density in kg/m3;
        s = 0.05  # rotor solidity;
        G = 0.503  # Rotor disc area in m2;
        U_tip = 120  # tip seep of the rotor blade(m/s);
        v_o = 4.3  # mean rotor induced velocity in hover;
        omega = 300  # blade angular velocity in radians/second;
        R = 0.4  # rotor radius in meter;
        delta = 0.012  # profile drage coefficient;
        k = 0.1  # incremental correction factor to induced power;
        W = 20  # aircraft weight in newton;
        P0 = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
        P1 = (1 + k) * (pow(W, (3 / 2)) / np.sqrt(2 * rho * G))
        P2 = 11.46
        t_n = t_n / 10

        x_pre = pre_l_n[0] * self.x_s + 0.5 * self.x_s
        y_pre = pre_l_n[1] * self.y_s + 0.5 * self.y_s
        z_pre = self.l_o_v + pre_h * self.h_s
        x = l_n[0] * self.x_s + 0.5 * self.x_s
        y = l_n[1] * self.y_s + 0.5 * self.y_s
        z = self.l_o_v + h * self.h_s

        d = np.sqrt((x_pre - x) ** 2 + (y_pre - y) ** 2)
        h = np.abs(z_pre - z)  #
        v_h = d / t_n  # 垂直速度
        v_v = h / t_n  # 水平速度
        Energy_uav = t_n * P0 * (1 + 3 * np.power(v_h, 2) / np.power(U_tip, 2)) + t_n * (
                1 / 2) * d_o * rho * s * G * np.power(v_h, 3) + \
                     t_n * P1 * np.sqrt(np.sqrt(1 + np.power(v_h, 4) / (4 * np.power(v_o, 4))) - np.power(v_h, 2) / (
                2 * np.power(v_o, 2))) + P2 * v_v * t_n
        return Energy_uav

    # 无人机的轨迹
    def UAV_FLY(self, UAV_trajectory_tmp, slot):
        UAV_trajectory = np.zeros((self.N_slot, 3))
        for slot in range(slot):
            UAV_trajectory[slot, 0] = UAV_trajectory_tmp[slot, 0] * self.x_s + 0.5 * self.x_s
            UAV_trajectory[slot, 1] = UAV_trajectory_tmp[slot, 1] * self.y_s + 0.5 * self.y_s
            UAV_trajectory[slot, 2] = self.l_o_v + UAV_trajectory_tmp[slot, 2] * self.h_s

        for slot in range(2, slot):
            diff = np.abs(UAV_trajectory[slot, 0] - UAV_trajectory[slot - 2, 0]) + np.abs(
                UAV_trajectory[slot, 1] - UAV_trajectory[slot - 2, 1])
            if (diff > self.x_s):
                UAV_trajectory[slot - 1, 0] = (UAV_trajectory[slot - 2, 0] + UAV_trajectory[slot, 0]) / 2
                UAV_trajectory[slot - 1, 1] = (UAV_trajectory[slot - 2, 1] + UAV_trajectory[slot, 1]) / 2
        return UAV_trajectory

    # 吞吐量和数据速率
    def throughput(self, UAV_trajectorys, UAV_flight_time, Task_offloadings, UE_Schedulings, Task_offloading_time, EP,
                   slot):
        through = np.zeros((EP, self.N_slot), dtype=np.float32)
        count = 0
        for ep in range(self.eps - EP, self.eps):  # self.eps=2000
            t_n = UAV_flight_time[ep, :]
            UAV_trajectory = UAV_trajectorys[ep, :]
            Task_offloading = Task_offloadings[ep, :]
            # 加一个用户的调度
            UE_Scheduling = UE_Schedulings[ep, :]
            # 加一个任务卸载时间
            t_n_t = Task_offloading_time[ep, :]
            for i in range(slot[0, ep]):
                h = UAV_trajectory[i, 2]
                x = UAV_trajectory[i, 0]
                y = UAV_trajectory[i, 1]
                for g in range(self.GTs):
                    if g == UE_Scheduling[i]:
                        for t in range(self.TKs):
                            a_kn = Task_offloading[i, t]
                            d = np.sqrt(mt.pow(h, 2) + mt.pow(x - self.w_k[g, 0], 2) + mt.pow(y - self.w_k[g, 1], 2))
                            if (np.sqrt(mt.pow(x - self.w_k[g, 0], 2) + mt.pow(y - self.w_k[g, 1], 2)) > 0):
                                ratio = h / np.sqrt(mt.pow(x - self.w_k[g, 0], 2) + mt.pow(y - self.w_k[g, 1], 2))
                            else:
                                ratio = np.Inf
                            p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
                            p_los = 1 / p_los
                            L_km = 20 * np.log10(d) + A * p_los + C
                            r_kn = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
                            through[count, i] = through[count, i] + t_n_t[i][t] * a_kn * (
                                    self.f_u * self.u_k[g, t, 0] * r_kn) / (
                                                        r_kn * self.u_k[g, t, 1] + self.f_u * self.u_k[
                                                    g, t, 0]) + self.f_g * (
                                                        1 - a_kn) * t_n[i]
            count = count + 1
        return through
