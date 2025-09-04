# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# steps = ['100k', '200k', '300k', '400k']
# x = np.arange(len(steps))

# # METRA data
# metra_means = [3.21, 3.66, 3.83, 3.99]
# metra_std = [0.71, 0.69, 0.90, 0.94]

# # Ours data
# ours_means = [3.51, 3.91, 4.34, 4.55]
# ours_std = [0.23, 0.43, 0.39, 0.35]

# # Plot setup
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot METRA with error bars
# ax.errorbar(x, metra_means, yerr=metra_std, fmt='-o', capsize=5, label='METRA', color='orange')

# # Plot Ours with error bars
# ax.errorbar(x, ours_means, yerr=ours_std, fmt='-o', capsize=5, label='Ours', color='blue')

# # X-axis labels and ticks
# ax.set_xticks(x)
# ax.set_xticklabels(steps)
# ax.set_xlabel('Steps', fontsize=12)

# # Y-axis label
# ax.set_ylabel('Performance', fontsize=12)

# # Title and legend
# ax.set_title('Kitchen Environment Performance (Skill Dimension=32)', fontsize=14)
# ax.legend(fontsize=12)

# # Grid for better readability
# ax.grid(True, linestyle='--', alpha=0.6)

# # Layout adjustment and display
# plt.tight_layout()
# plt.savefig('kitchen_performance0.png', dpi=300)


import matplotlib.pyplot as plt
import numpy as np

# Data
steps = ['100k', '200k', '300k', '400k']
x = np.arange(len(steps))

# Data for Skill Dimension=24
metra_means = [2.72, 3.20, 3.93, 3.94]
metra_std = [0.19, 0.27, 0.67, 0.36]

ours_means = [2.41, 3.61, 4.29, 5.08]
ours_std = [0.02, 0.02, 0.11, 0.45]

# Plot setup
fig, ax = plt.subplots(figsize=(8, 5))

# Plot METRA data with error bars
ax.errorbar(x, metra_means, yerr=metra_std, fmt='-o', capsize=5, label='METRA', color='orange')

# Plot Ours data with error bars
ax.errorbar(x, ours_means, yerr=ours_std, fmt='-o', capsize=5, label='Ours', color='blue')

# X-axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(steps)
ax.set_xlabel('Training Steps', fontsize=12)

# Y-axis label
ax.set_ylabel('Performance', fontsize=12)

# Title and legend
ax.set_title('Kitchen Environment Performance (Skill Dimension=24)', fontsize=14)
ax.legend(fontsize=12)

# Grid for clarity
ax.grid(True, linestyle='--', alpha=0.6)

# Layout adjustment and display
plt.tight_layout()
plt.savefig('kitchen_performance24.png', dpi=300)
