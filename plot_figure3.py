import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from paths import *

# manually copied from the log files
precision = """
s40000 b16 1.0236839387281705e-06
s30000 b8 5.617506285357106e-07
s30000 b32 5.179893452075327e-07
s30000 b4 5.063235391271304e-07
s30000 b2 5.001541122158726e-07
s20000 b32 4.571325434997682e-07
s10000 b1 4.055648440893784e-07
s50000 b1 8.486953829810722e-07
s40000 b32 1.5175776499196796e-06
s50000 b16 8.344162044930459e-07
s30000 b1 5.46869530122963e-07
s10000 b32 3.459826499239953e-07
s10000 b8 3.2734909180933025e-07
s10000 b4 3.3816304421272675e-07
s50000 b32 9.035036777067e-07
s10000 b64 3.607602933868194e-07
s20000 b16 5.333502401642545e-07
s30000 b16 5.267952675803647e-07
s40000 b4 1.0959549829598282e-06
s10000 b2 5.063599002524331e-07
s40000 b64 1.5065638415121861e-06
s50000 b4 8.386167360615594e-07
s20000 b64 4.2327545187146416e-07
s30000 b64 4.837603979397202e-07
s20000 b1 4.6159322732069017e-07
s40000 b8 9.84094567100812e-07
s40000 b2 8.963879449340116e-07
s50000 b2 7.566855832270823e-07
s20000 b4 5.705122802364005e-07
s20000 b2 4.564040974194915e-07
s20000 b8 5.100722664990669e-07
s50000 b8 8.122480697355498e-07
s40000 b1 1.6023307054065682e-06
s50000 b64 7.443724365390295e-07
s10000 b16 3.713149666511294e-07
"""

precision = [e for e in precision.split('\n') if len(e) > 0]
precision = [e.split(' ') for e in precision]
precision = [[float(i) for i in [e[0][1:], e[1][1:], e[2]]] for e in precision]

precision = sorted(precision, key=lambda x: x[0] + x[1], reverse=False)
print(precision)
precision = np.array(precision)

precision_fed_svd = precision[:, -1].reshape([-1, 5])

plot_label = ['FedSVD b=%s' % (2 ** e) for e in range(len(precision_fed_svd))]

time_x = [10000, 20000, 30000, 40000, 50000]

fig, ax = plt.subplots(1, 1, figsize=[7, 4])

for i in range(len(precision_fed_svd)):
    ax.plot(time_x, precision_fed_svd[i], label=plot_label[i], marker='*')


ax.set_ylim(1e-8, 5e-6)
ax.set_xlabel('n')
ax.set_xticks(time_x)
ax.set_xticklabels(['10k', '20k', '30k', '40k', '50k'])
ax.set_ylabel('Mean Absolute Percentage Error (MAPE)')
ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=6))
ax.legend()

fig.tight_layout()
plt.savefig(os.path.join(images_dir, 'precision_mape.png'), type="png", dpi=300)
plt.show()
