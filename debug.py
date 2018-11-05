import matplotlib.pyplot as plt
import numpy as np


with open('data/map_data.txt', 'r') as f:
    landmarks = [l.strip().split('\t')for l in f.readlines()]

with open('data/obsv1.txt', 'r') as f:
    obsv = [l.strip().split(' ') for l in f.readlines()]

landmarks = np.array(landmarks, dtype=np.float32)
obsv = np.array(obsv, dtype=np.float32)

plt.scatter(landmarks[:, 0], landmarks[:, 1])
plt.scatter(obsv[:, 0], obsv[:, 1], marker='+')
plt.show()
