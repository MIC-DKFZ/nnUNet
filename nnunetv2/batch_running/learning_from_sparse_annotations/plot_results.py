import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nnunetv2.utilities.overlay_plots import color_cycle, hex_to_rgb

sns.set(rc={'figure.figsize':(15, 10)})
a = np.loadtxt('/home/isensee/Downloads/ablation studies nnUNet - deletenem (6).tsv', delimiter='\t', dtype=str, skiprows=0)
x = a[:, 2].astype(float)
tmp = np.argsort(x)
y = a[:, 3].astype(float)
hue = a[:, 4]
x = x[tmp]
y = y[tmp]
hue = hue[tmp]
legend = []
types = np.unique(hue)
for i, t in enumerate(types):
    idx = hue == t
    plt.plot(x[idx], y[idx], color=[i / 255 for i in hex_to_rgb(color_cycle[i])], marker='o', ms=12, linewidth=4)
    legend.append(t)
plt.legend(legend)
plt.xlabel('percent foreground annotated')
plt.ylabel('Dice')
plt.hlines(0.8718, 0, 0.51)
plt.xlim((0, 0.51))
# sns.scatterplot(x=x, y=y, hue=hue, palette='deep', s=100)
plt.savefig('tmp.png')
plt.close()



# some eval of trainer performance
a = np.loadtxt('/home/isensee/Documents/tmp.csv', delimiter=',', dtype=str, skiprows=0)
classes = np.unique(a[:, 0])
for c in classes:
    pos = a[:, 0] == c
    mn = a[:, 1][pos].astype(float).mean()
    print(c, mn)