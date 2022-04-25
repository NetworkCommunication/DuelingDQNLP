import matplotlib.pyplot as plt

if __name__ == '__main__':
    x0 = [1 - 0.2, 2 - 0.2, 3 - 0.2, 4 - 0.2, 5 - 0.2]
    x1 = [1, 2, 3, 4, 5]
    x2 = [1.2, 2.2, 3.2, 4.2, 5.2]
    y1 = [5.9, 4.0, 3.9, 2.8, 1.9]
    y2 = [0.773288, 0.777036, 0.783868, 0.801638, 0.758817]
    ax1 = plt.figure().subplots()
    line1 = ax1.bar(x0, y1, width=0.4, color="#F5542A", alpha=0.8)
    ax1.set_xlabel('Size of the Segment')
    ax1.set_ylabel('Average Runtime Per Episode(s)')
    ax1.set_ylim(1, 7)
    ax2 = ax1.twinx()
    line2 = ax2.bar(x2, y2, width=0.4, color="#70B2DE", alpha=0.8)
    ax2.set_ylabel('Energy Efficiency')
    ax2.set_ylim(0.5, 0.9)
    plt.xticks(x1, [0.1, 0.2, 0.4, 0.5, 1])
    plt.legend(handles=[line1, line2], labels=['Average Runtime Per Episode', 'Energy Efficiency'],
               loc="upper right")  # 设置折线名称  # 设置折线名称
    plt.grid()
    plt.show()  # 显示折线图
