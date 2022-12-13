import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nnunetv2.utilities.overlay_plots import color_cycle, hex_to_rgb

sns.set(rc={'figure.figsize':(15, 10)})
a = np.loadtxt('/home/isensee/Downloads/ablation studies nnUNet - deletenem.tsv', delimiter='\t', dtype=str, skiprows=2)
x = a[:, 2].astype(float)
y = a[:, 3].astype(float)
hue = a[:, 4]
legend = []
types = np.unique(hue)
for i, t in enumerate(types):
    idx = hue == t
    plt.plot(x[idx], y[idx], color=[i / 255 for i in hex_to_rgb(color_cycle[i])], marker='o', ms=12, linewidth=4)
    legend.append(t)
plt.legend(legend)
plt.xlabel('percent foreground annotated')
plt.ylabel('Dice')
# sns.scatterplot(x=x, y=y, hue=hue, palette='deep', s=100)
plt.savefig('tmp.png')
plt.close()

