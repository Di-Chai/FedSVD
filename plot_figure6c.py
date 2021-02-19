import os
import numpy as np
import matplotlib.pyplot as plt

from paths import *
from utils import parse_log_params


def parse_time(string):
    return float(string.split(' ')[-2])


log_dir = os.path.join(log_dir, 'comm')

efficiency_logs = []
for e in os.listdir(log_dir):
    if e.endswith('.log'):
        with open(os.path.join(log_dir, e), 'r') as f:
            efficiency_logs.append(f.readlines())

efficiency_logs = sorted(
    efficiency_logs,
    key=lambda x: int(parse_log_params(x[0])['num_samples'])*int(parse_log_params(x[0])['num_participants']) +
                  int(parse_log_params(x[0])['num_participants'] * 1000)
)

communication = [float(e[10].strip('\n').split(' ')[-1]) for e in efficiency_logs]

communication = np.array(communication).reshape([-1, 5]).T

x = [2, 4, 6, 8, 10, 12, 14, 16]
label = ['10k', '20k', '30k', '40k', '50k']
label = ['n=' + e for e in label]

fig, ax = plt.subplots(1, 1, figsize=[6, 4])

for i in range(len(communication)):
    ax.plot(x, communication[i], label=label[i], marker='*')

ax.set_title('Amount of Communication Data (per data holder)')
ax.set_xlabel('# of data holders')
ax.set_xticks(x)
ax.set_ylabel('Communication Data (MB)')
ax.legend()

plt.savefig(os.path.join(images_dir, 'scalability3.png'), type="png", dpi=300)
plt.show()
