import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.0, 1.0, 0.1)
y_1 = np.asarray([0.03, 0.05, 0.07, 0.11, 0.12, 0.14, 0.18, 0.20, 0.24, 0.28])
y_2 = np.asarray([0.1, 0.2, 0.23, 0.5, 0.57, 0.63, 0.8, 0.82, 0.84, 0.86])

plt.plot(x, y_1, label="Annotation ratio")
plt.plot(x, y_2, label="Dice")
plt.show()