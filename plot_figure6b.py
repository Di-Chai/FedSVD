import os
import numpy as np
import matplotlib.pyplot as plt

from paths import *
from utils import parse_log_params


def parse_time(string):
    return float(string.split(' ')[-2])


log_dir = os.path.join(log_dir, 'scalability')

efficiency_logs = []
for e in os.listdir(log_dir):
    with open(os.path.join(log_dir, e), 'r') as f:
        efficiency_logs.append(f.readlines())

efficiency_logs = sorted(
    efficiency_logs,
    key=lambda x: int(parse_log_params(x[0])['num_samples'])
)

time = [
    [float(e[1].strip('\n').split(' ')[-1]),
     parse_time(e[4]) + parse_time(e[5]),
     parse_time(e[6]), parse_time(e[7]), parse_time(e[8])]
    for e in efficiency_logs
]

time_plot = [
    [np.array([e[0] for e in time]).reshape([-1, 5]).mean(1), 'Standalone SVD'],
    [np.array([np.sum(e[1:]) for e in time]).reshape([-1, 5]).mean(1), 'FedSVD b=N/1000'],
]

ratio = time_plot[1][0] / time_plot[0][0]

time_x = list(range(len(time_plot[0][0])))

fig, ax = plt.subplots(1, 1, figsize=[6, 4])

for plot in time_plot:
    ax.plot(time_x, plot[0], label=plot[1], marker='*')

ax_twin = ax.twinx()
ax_twin.plot(time_x, ratio, label='Ratio (FedSVD/Standalone)', marker='.', color='g')
ax_twin.legend(loc='upper left')
ax_twin.set_ylabel('Ratio')
ax_twin.set_ylim(1, 2)

ax.set_yscale('log')
ax.set_ylim(0, 10**4)
ax.set_title('')
ax.set_xlabel('n')
ax.set_xticks(time_x)
ax.set_xticklabels(['10k', '20k', '30k', '40k', '50k', '60k', '70k', '80k', '90k', '100k'])
ax.set_ylabel('Time (Seconds)')
ax.legend()

# plt.show()
plt.savefig(os.path.join(images_dir, 'scalability2.png'), type="png", dpi=300)
