
# coding: utf-8

# In[34]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

get_ipython().magic('matplotlib inline')
# notebook
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (32.0, 24.0)
pylab.rcParams['font.size'] = 40
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = 'Computer Modern'


# In[35]:


fig, ax = plt.subplots()
ax.axis([0.01, 100, 0.01, 1000])
ax.loglog()

ax.set_xlabel('Operational Intensity (FLOP/byte)')
ax.set_ylabel('Single precision GFLOP/s (SIMD)')

# ax.set_aspect(1/1.6)

ax.plot([0.01,0.4,100], [18,665.6,665.6], linestyle='solid', c="#b3d9ff", linewidth=5, label="L1 Cache Bandwidth = 1551.76 GB/s")
ax.plot([0.01,0.9,100], [8,665.6,665.6], linestyle='solid', c="#66b3ff", linewidth=5, label="L2 Cache Bandwidth = 780.11 GB/s")
ax.plot([0.01,1.6,100], [4.1,665.6,665.6], linestyle='solid', c="#0080ff", linewidth=5, label="L3 Cache Bandwidth = 418.32 GB/s")
ax.plot([0.01,10,100], [0.65,665.6,665.6], linestyle='solid', c="#004d99", linewidth=5, label="DRAM Bandwidth = 65.96 GB/s")
ax.scatter(1.69, 9.97, s=600, zorder=3, label="1024 Serial = 9.97 GFLOP/s")
ax.scatter(1.69, 65.01,s=600, zorder=4, label="1024 OpenMP = 65.01 GFLOP/s")
ax.scatter(2.15, 35.69,s=600, zorder=5, label="1024 OpenCL = 35.69 GFLOP/s")
ax.scatter(1.69, 32.33,s=600, zorder=6, label="1024 MPI = 32.33 GFLOP/s")
ax.plot([1.69,1.69], [0,1000], linestyle='dashed', c='black', alpha=0.4, zorder=1, linewidth=5, label="OI = 1.69")
ax.plot([2.15,2.15], [0,1000], linestyle='dashdot', c='black', alpha=0.4, zorder=2, linewidth=5, label="OI = 2.15")
ax.legend(loc="lower right", frameon=True, shadow=None, fancybox=False, edgecolor='white', framealpha=0.9)

plt.setp(ax.spines.values(), linewidth=4)

# Pad the ticks on both the axis so the bottom left corner does not overlap
pylab.rcParams['xtick.major.pad']='15'
pylab.rcParams['ytick.major.pad']='15'

# Change the ticks from scientific notation to standarn notation
from matplotlib.ticker import ScalarFormatter
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.savefig('cpu-roofline.pdf')
plt.show()


# In[36]:


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


# In[37]:


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


# In[46]:


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

ax.bar(y_pos, performance, align='center', alpha=1, width=1.0, edgecolor='black', linewidth=4, color=colors)
plt.xlim([-0.5,y_pos.size-0.5])
plt.xticks([])

plt.setp(ax.spines.values(), linewidth=4)

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
    cell.set_linewidth(4)

plt.setp(ax.get_yticklabels()[0], visible=False) 

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.savefig('autotune.png')
plt.show()


# In[45]:


fig, ax = plt.subplots()

ax.set_ylabel('Runtime (s)')

objects = ('4' , '4' , '4' , '4' , '4' , '4', '4', '4',
           '8' , '8' , '8' , '8' , '8' , '8', '4', '4')

y_pos = np.arange(len(objects))

performance = [28.45,23.23,19.15,4.02,3.45,3.20,2.94,3.95,17.30,14.82,10.12,3.91,3.39,3.05,2.93,2.89]

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

ax.bar(y_pos, performance, align='center', alpha=1, width=1.0, edgecolor='black', linewidth=4, color=colors)
plt.xlim([-0.5,y_pos.size-0.5])
plt.xticks([])

plt.setp(ax.spines.values(), linewidth=4)

rows = ('X Size', 'Y Size')

table_text = [['1' , '2' , '4' , '8' , '16', '32', '64', '128',
               '1' , '2' , '4' , '8' , '16', '32', '64', '128'],
              ['1', '1', '1', '1', '1', '1', '1', '1',
               '2', '2', '2', '2', '2', '2', '2', '2']]

# Add a table at the bottom of the axes
the_table = plt.table(cellText=table_text,
                      rowLabels=rows,
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0, -0.10, 1, 0.10])

for key, cell in the_table.get_celld().items():
    cell.set_linewidth(4)

plt.setp(ax.get_yticklabels()[0], visible=False) 

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.show()


# In[40]:


def calc_band(size, runtime, iterations):
    return (size*size*9*4*2*iterations/runtime/1000000000)

print(calc_band(1024, 39.3, 20000))

fig, ax = plt.subplots()

xs = [64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024,1088,
      1152,1216,1280,1344,1408,1472,1536,1600,1664,1728,1792,1856,1920]

runtimes = [0.48,0.79,1.38,2.12,3.09,4.27,5.67,7.28,9.11,11.28,13.41,15.89,18.56,21.57,24.60,27.91,
            31.47,35.23,39.21,43.62,47.80,52.42,57.24,62.30,67.56,73.04,78.74,84.66,90.82,97.19]

ys = []

for i in range(len(xs)):
    ys.append(calc_band(xs[i], runtimes[i], 40000));

ax.plot(xs, ys, c="black", linewidth=7)

plt.setp(ax.spines.values(), linewidth=5)

plt.show()


# In[41]:


fig, ax = plt.subplots()
ax.axis([0.01, 1000, 0.01, 10000])
ax.loglog()

ax.set_xlabel('Operational Intensity (FLOP/byte)')
ax.set_ylabel('Single precision GFLOP/s (SIMD)')

# ax.set_aspect(1/1.6)

ax.plot([0.01,23.3,1000], [2,3524,3524], linestyle='solid', c="#0080ff", linewidth=5, label="Peak Bandwidth = 151.00 GB/s")
ax.scatter(2.15, 133.49, c="#ff6666", s=600, label="128 = 133.49 GFLOP/s")
ax.scatter(2.15, 194.17, c="#cc0000",s=600, label="256 = 194.17 GFLOP/s")
ax.scatter(2.15, 243.72, c="#800000",s=600, label="1024 = 243.72 GFLOP/s")
ax.plot([2.15,2.15], [0,10000], linestyle='dashed', c='black', alpha=0.4, zorder=0, linewidth=5, label="OI = 2.15")
ax.legend(loc="lower right", frameon=False)

plt.setp(ax.spines.values(), linewidth=4)

# Pad the ticks on both the axis so the bottom left corner does not overlap
pylab.rcParams['xtick.major.pad']='15'
pylab.rcParams['ytick.major.pad']='15'

# Change the ticks from scientific notation to standarn notation
from matplotlib.ticker import ScalarFormatter
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.savefig('gpu-roofline.pdf')
plt.show()

