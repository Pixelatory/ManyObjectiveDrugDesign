import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from build_vocab import frags_above_freq_thresh

with open('MTLBERT/DeepFMPO/python3/Modules/allmolgen_frag_freq.pkl', 'rb') as f:
    frag_freq = pickle.load(f)

df = pd.DataFrame(list(frag_freq.items()), columns=['frag', 'Frequency'])
print(df)

fig, ax = plt.subplots()
displot = sns.ecdfplot(df, x="Frequency", complementary=True, stat='proportion', log_scale=True, ax=ax)

#ax.set_xlim(left=1)  # Set the lower bound to 1
x_values = ax.lines[0].get_xdata()
y_values = ax.lines[0].get_ydata()
mask = x_values >= -1000
plt.fill_between(x_values, y_values, where=mask, color='blue', alpha=0.2)

#plt.axvline(5000, c='k')
#plt.text(5500, 0.8, f"{len(df[df['Frequency'] >= 5000]) / len(df) * 100:.1f}% (5000)", rotation=90)

#plt.axvline(2000, c='k')
#plt.text(2200, 0.8, f"{len(df[df['Frequency'] >= 2000]) / len(df) * 100:.1f}% (2000)", rotation=90)

plt.axvline(1000, c='k')
plt.text(1100, 0.7, f"{len(df[df['Frequency'] >= 1000]) / len(df) * 100:.1f}% (1000)", rotation=90)

plt.axvline(500, c='k')
plt.text(550, 0.7, f"{len(df[df['Frequency'] >= 500]) / len(df) * 100:.1f}% (500)", rotation=90)

plt.axvline(100, c='k')
plt.text(110, 0.7, f"{len(df[df['Frequency'] >= 100]) / len(df) * 100:.1f}% (100)", rotation=90)

plt.axvline(10, c='k')
plt.text(11, 0.7, f"{len(df[df['Frequency'] >= 10]) / len(df) * 100:.1f}% (10)", rotation=90)

plt.axvline(5, c='k')
plt.text(5.4, 0.7, f"{len(df[df['Frequency'] >= 5]) / len(df) * 100:.1f}% (5)", rotation=90)

plt.axvline(2, c='k')
plt.text(2.2, 0.7, f"{len(df[df['Frequency'] >= 2]) / len(df) * 100:.1f}% (2)", rotation=90)

#ax.ax.fill_between(x1[line1sec], y1[line1sec], color="tab:blue", alpha=0.3)
props = dict(boxstyle='round', facecolor='tab:blue', alpha=0.5)
plt.text(12000, 0.5, "Total Fragments: \n67860", multialignment='center', verticalalignment='top', bbox=props)
plt.tight_layout(h_pad=2.0)
plt.savefig('tmp.png')
plt.show()