
# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

get_ipython().magic('matplotlib inline')
# notebook
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (32.0, 24.0)
pylab.rcParams['font.size'] = 40
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = 'Computer Modern'


# In[2]:


fig, ax = plt.subplots()
ax.axis([0.01, 100, 0.1, 1000])
ax.loglog()

ax.set_xlabel('Operational Intensity (FLOPS/byte)')
ax.set_ylabel('Single precision GFLOP/s (SIMD)')

# ax.set_aspect(1/1.6)

ax.plot([0.01,0.4,100], [18,665.6,665.6], linestyle='solid', c="#b3d9ff", linewidth=7, label="L1 Cache Bandwidth = 1551.76 GB/s")
ax.plot([0.01,0.9,100], [8,665.6,665.6], linestyle='solid', c="#66b3ff", linewidth=7, label="L2 Cache Bandwidth = 780.11 GB/s")
ax.plot([0.01,1.6,100], [4.1,665.6,665.6], linestyle='solid', c="#0080ff", linewidth=7, label="L3 Cache Bandwidth = 418.32 GB/s")
ax.plot([0.01,10,100], [0.65,665.6,665.6], linestyle='solid', c="#004d99", linewidth=7, label="DRAM Bandwidth = 65.96 GB/s")
ax.scatter(0.34, 96.81, c="#ff6666", s=600, label="128 = 96.81 GFLOP/s")
ax.scatter(0.34, 139.84, c="#cc0000",s=600, label="256 = 139.84 GFLOP/s")
ax.scatter(0.34, 62.96, c="#800000",s=600, label="1024 = 62.96 GFLOP/s")
ax.legend(loc="lower right", frameon=False)

plt.setp(ax.spines.values(), linewidth=5)

# Pad the ticks on both the axis so the bottom left corner does not overlap
pylab.rcParams['xtick.major.pad']='15'
pylab.rcParams['ytick.major.pad']='15'

# Change the ticks from scientific notation to standarn notation
from matplotlib.ticker import ScalarFormatter
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

plt.show()


# In[3]:


fig, ax = plt.subplots()

ax.set_xlabel('Threads')
ax.set_ylabel('Time (s)')
ax.set_xlim([1,16])
# ax.set_ylim([0,6.4])

# ax.set_aspect(1.5)

# Make sure that all cores show up in the x axis
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

# 128x128 OMP_PROC_BIND=spread
ax.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 
        [6.33, 3.58, 2.46, 1.88, 1.58, 1.40, 1.23, 1.13, 1.08, 0.98, 0.98, 0.90, 0.87, 0.84, 0.84, 0.81], linewidth=7, label="OMP_PROC_BIND=spread")

# 128x128 OMP_PROC_BIND=close
ax.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 
        [6.35, 3.26, 2.28, 1.73, 1.46, 1.25, 1.12, 0.98, 1.10, 0.99, 0.91, 0.90, 0.86, 0.83, 0.81, 0.80], linewidth=7, label="OMP_PROC_BIND=close")

ax.legend(loc="upper right", frameon=False)

plt.setp(ax.spines.values(), linewidth=5)

plt.show()


# In[4]:


def calc_bandwidth(size, runtime, flops, iters, intensity):
    ans = ((size*size*flops*iters)/runtime)/intensity/1000000000
    return ans

fig, ax = plt.subplots()

ax.set_xlabel('Array Size ($2^x$ floats)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_xlim([1,15])
ax.set_ylim([0,700])

# send_recv_stencil.c 1024
ax.plot([1,4,5,6,7,8,9,10,11,12,13,14,15], 
        [calc_bandwidth(2, 0.000420, 9, 200, 0.375),
         calc_bandwidth(16, 0.000858, 9, 200, 0.375),
         calc_bandwidth(32, 0.000862, 9, 200, 0.375),
         calc_bandwidth(64, 0.000907, 9, 200, 0.375),
         calc_bandwidth(128, 0.001039, 9, 200, 0.375),
         calc_bandwidth(256, 0.001473, 9, 200, 0.375),
         calc_bandwidth(512, 0.003291, 9, 200, 0.375),
         calc_bandwidth(1024, 0.009317, 9, 200, 0.375),
         calc_bandwidth(2048, 0.036165, 9, 200, 0.375),
         calc_bandwidth(4096, 0.543365, 9, 200, 0.375),
         calc_bandwidth(8000, 2.007064, 9, 200, 0.375),
         calc_bandwidth(16000, 7.788455, 9, 200, 0.375),
         calc_bandwidth(32000, 31.167855, 9, 200, 0.375)], c="black", linewidth=7)

ax.scatter([1,4,5,6,7,8,9,10,11,12,13,14,15], 
        [calc_bandwidth(2, 0.000420, 9, 200, 0.375),
         calc_bandwidth(16, 0.000858, 9, 200, 0.375),
         calc_bandwidth(32, 0.000862, 9, 200, 0.375),
         calc_bandwidth(64, 0.000907, 9, 200, 0.375),
         calc_bandwidth(128, 0.001039, 9, 200, 0.375),
         calc_bandwidth(256, 0.001473, 9, 200, 0.375),
         calc_bandwidth(512, 0.003291, 9, 200, 0.375),
         calc_bandwidth(1024, 0.009317, 9, 200, 0.375),
         calc_bandwidth(2048, 0.036165, 9, 200, 0.375),
         calc_bandwidth(4096, 0.543365, 9, 200, 0.375),
         calc_bandwidth(8000, 2.007064, 9, 200, 0.375),
         calc_bandwidth(16000, 7.788455, 9, 200, 0.375),
         calc_bandwidth(32000, 31.167855, 9, 200, 0.375)], c="black", s=900, marker='x')

plt.setp(ax.spines.values(), linewidth=5)

# from matplotlib.ticker import FormatStrFormatter

# ax.yaxis.set_major_formatter(FormatStrFormatter('%2.2f'))

plt.show()


# In[146]:


fig, ax = plt.subplots()

ax.set_ylabel('Runtime (s)')

objects = ('4' , '4' , '4' , '4' , '4' , '4' ,
           '8' , '8' , '8' , '8' , '8' , '8' ,
           '16', '16', '16', '16', '16', '16',
           '32', '32', '32', '32', '32', '32')

y_pos = np.arange(len(objects))

performance = [5.72,3.12,1.75,1.09,0.79,0.76,
               3.90,2.20,1.36,0.89,0.78,0.77,
               3.22,1.93,1.16,0.85,0.80,0.90,
               3.06,1.80,1.12,0.86,0.91,1.29]

colors = []

maximum = max(performance)
minimum = min(performance)

performance_range = maximum - minimum

for i in performance:
    normalised = (i - minimum) / performance_range
    r = 0.95 - (normalised / 4)
    g = 0.95 - (normalised / 4)
    b = 0.95 - (normalised / 4)
    colors.append((r, g, b))

ax.bar(y_pos, performance, align='center', alpha=1, width=1.0, edgecolor='black', linewidth=5, color=colors)
plt.xlim([-0.5,y_pos.size-0.5])
plt.xticks([])

plt.setp(ax.spines.values(), linewidth=5)

rows = ('X Size', 'Y Size')

table_text = [['1' , '2' , '4' , '8' , '16', '32',
               '1' , '2' , '4' , '8' , '16', '32', 
               '1' , '2' , '4' , '8' , '16', '32',
               '1' , '2' , '4' , '8' , '16', '32'],
              ['4' , '4' , '4' , '4' , '4' , '4' ,
               '8' , '8' , '8' , '8' , '8' , '8' ,
               '16', '16', '16', '16', '16', '16',
               '32', '32', '32', '32', '32', '32']]

# Add a table at the bottom of the axes
the_table = plt.table(cellText=table_text,
                      rowLabels=rows,
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0, -0.10, 1, 0.10])

for key, cell in the_table.get_celld().items():
    cell.set_linewidth(5)

plt.setp(ax.get_yticklabels()[0], visible=False) 

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.show()

