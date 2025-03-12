# Fonts

from matplotlib import pyplot as plt
import scienceplots

print('Using fonts from imported file')

# plt.style.use('seaborn-v0_8-paper')
# plt.style.use('ggplot')
plt.style.use(['science','nature'])
# plt.style.use('seaborn-v0_8-pastel')
# plt.xkcd(scale=2, length=100, randomness=2)


# Extract the color cycle from 'ggplot'
with plt.style.context('ggplot'):
    ggplot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Update the current color cycle to use ggplot colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=ggplot_colors)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r'\usepackage{amsmath}'  # For math symbols
     })

TINY_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

