import numpy as np
import matplotlib.pyplot as plt

with open('data/RSC_awake.txt', 'r') as f:
    awake = f.readlines()

with open('data/RSC_asleep.txt', 'r') as f:
    asleep = f.readlines()

awake = [float(x.split('|')[-1].split(':')[-1]) for x in awake]
asleep = [float(x.split('|')[-1].split(':')[-1]) for x in asleep]
epochs = np.arange(100)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(epochs, awake, c='tab:orange', label='model-wake')
ax.plot(epochs, asleep, c='k', label='model-sleep')
ax.set_xlabel('Epochs')
ax.set_ylabel(r'$-f(x | \theta)$')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.grid()
ax.legend()
plt.savefig('figures/training_curves_rsc.png', bbox_inches='tight', dpi=100)
plt.show()
