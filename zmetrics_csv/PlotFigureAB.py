import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import pandas as pd


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



import matplotlib.pyplot as plt
import numpy as np

# 数据

species = ("alpha_unseen", "alpha_seen", "window_size")
penguin_means = {
    '1': (62, 56, 57),
    '2': (66, 66, 66), 
    '3': (62.7, 61, 67),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 1

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

plt.tick_params(axis='both', labelsize=20)
# plt.xlabel(x_label, font=TimesPath, fontsize=40)
plt.ylabel('CoverCoords', fontsize=40)
# plt.title(env_name, font=TimesPath, fontsize=40, pad=15)
# plt.legend(fontsize=60, prop=font_prop, loc='lower right')
# plt.subplots_adjust(bottom=0.15)  
# plt.subplots_adjust(top=0.9)
plt.xticks([])
# plt.grid(True)

plt.savefig('1.png')
