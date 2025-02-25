# Fonts

from matplotlib import pyplot as plt

print('Using fonts from imported file')

# plt.style.use('seaborn-v0_8-paper')
plt.style.use('ggplot')
# plt.style.use('seaborn-v0_8-pastel')

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Computer Modern"],
    "text.latex.preamble": r'\usepackage{amsmath}'  # For math symbols
     })

TINY_SIZE = 10
SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIG_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)     # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

