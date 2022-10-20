# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 11:34
# @Author  : 
# @File    : draw_fig.py

import numpy as np
import matplotlib.pyplot as plt

# tabu_length = [0, 5, 15, 50]
neighbor_range = [[50, 100], [100, 100], [100, 200], [200, 200]]
color = ['salmon', 'sandybrown', 'greenyellow', 'darkturquoise']
fig = plt.figure()
ax_1 = fig.add_subplot(211)
# ax_1.set_aspect(1)
ax_1.set(ylabel='Average traveling costs', xlim=[0, 3000], ylim=[420, 600])
plt.grid(linestyle='--', linewidth=1, alpha=0.3)

ax_2 = fig.add_subplot(212)
# ax_2.set_aspect(1.2)
ax_2.set(xlabel='No. of generation', ylabel='Best so far',
         xlim=[0, 3000], ylim=[420, 550])
plt.grid(linestyle='--', linewidth=1, alpha=0.3)
# fig.tight_layout()
fig.suptitle('Tuning TABU_LENGTH', fontweight="bold")

for _i, value in enumerate(neighbor_range):
    # _m = np.load("tuning_tabu_length/mean_fitness_{}.npy".format(_i))
    # _b = np.load("tuning_tabu_length/best_fitness_{}.npy".format(_i))
    _m = np.load("tuning_neighbor_range/mean_fitness_{}.npy".format(_i))
    _b = np.load("tuning_neighbor_range/best_fitness_{}.npy".format(_i))

    mean = _m.tolist()
    best = _b.tolist()

    x_axis = len(mean)
    x_axis = list(range(x_axis))

    # ax_1.plot(x_axis, mean, color=color[_i], label='{}'.format(value), ls='-')
    # ax_2.plot(x_axis, best, color=color[_i], label='{}'.format(value), ls='-')
    ax_1.plot(x_axis, mean, color=color[_i], label='{}/{}'.format(value[0], value[1]), ls='-')
    ax_2.plot(x_axis, best, color=color[_i], label='{}/{}'.format(value[0], value[1]), ls='-')

ax_1.legend(loc='upper right')
ax_2.legend(loc='upper right')
fig.savefig('../../fig/[tmp]Tuning NEIGHBOR_RANGE.pdf')
# fig.show()


