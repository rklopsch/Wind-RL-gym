# Fonts

from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8-paper')
# plt.style.use('seaborn-v0_8-pastel')

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Computer Modern"],
    "text.latex.preamble": r'\usepackage{amsmath}'  # For math symbols
     })

TINY_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIG_SIZE = 20
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)     # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

