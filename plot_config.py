import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, cm
# global settings of plots DO NOT CHANGE
medium_fontsize = 20.5
rcParams['text.latex.preamble']= r"""
\usepackage{amsmath}
\boldmath
"""
font = {'size': medium_fontsize, 'family': 'sans-serif', 'weight': 'bold'}
rc('font', **font)
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['lines.linewidth'] = 2.5
rcParams['figure.figsize'] = (8, 8)
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.labelsize'] = medium_fontsize
rc('text', usetex=False)
