from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

from math import sin
from math import cos
from math import radians

#--- FUNCTIONS ----------------------------------------------------------------+

def plot_organism(x1, y1, theta, ax):

    circle = Circle([x1,y1], 0.05, edgecolor = 'g', facecolor = 'lightgreen', zorder=8)
    ax.add_artist(circle)

    edge = Circle([x1,y1], 0.05, facecolor='None', edgecolor = 'darkgreen', zorder=8)
    ax.add_artist(edge)

    tail_len = 0.075
    
    x2 = cos(radians(theta)) * tail_len + x1
    y2 = sin(radians(theta)) * tail_len + y1

    ax.add_line(lines.Line2D([x1,x2],[y1,y2], color='darkgreen', linewidth=1, zorder=10))

    pass


def plot_food(x1, y1, ax):

    circle = Circle([x1,y1], 0.03, edgecolor = 'darkslateblue', facecolor = 'mediumslateblue', zorder=5)
    ax.add_artist(circle)
    
    pass

#--- END ----------------------------------------------------------------------+
