import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, rc, rcParams

# global settings of plots DO NOT CHANGE
medium_fontsize = 20.5
rcParams[
    "text.latex.preamble"
] = r"""
\usepackage{amsmath}
\boldmath
"""
font = {"size": medium_fontsize, "family": "sans-serif", "weight": "bold"}
rc("font", **font)
rcParams["axes.linewidth"] = 1.5
rcParams["xtick.major.width"] = 1.5
rcParams["ytick.major.width"] = 1.5
rcParams["lines.linewidth"] = 2.5
rcParams["figure.figsize"] = (8, 8)
rcParams["axes.labelweight"] = "bold"
rcParams["axes.labelsize"] = medium_fontsize
rc("text", usetex=False)
plt.axes.labelweight = "bold"

# Set a fixed figure size
figsize = (12, 8)
rcParams["figure.figsize"] = figsize
