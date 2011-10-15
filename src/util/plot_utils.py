__author__="huziy"
__date__ ="$Aug 17, 2011 3:49:28 PM$"


import numpy as np

def apply_plot_params(font_size = 20, width_pt = 1000, aspect_ratio = 1):
    """
    aspect_ratio = height / (width * golden_mean)
    """
    import pylab
    import math
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (math.sqrt(5.0) - 1.0) / 2.0       # Aesthetic ratio
    fig_width = width_pt * inches_per_pt          # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width,  aspect_ratio * fig_height]

    params = {
        'axes.labelsize': font_size,
        'font.size':font_size,
        'text.fontsize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': fig_size
        }

    pylab.rcParams.update(params)

def zoom_to_qc(plotter = None):
    ymin, ymax = plotter.ylim()
    plotter.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)

    xmin, xmax = plotter.xlim()
    plotter.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)


def draw_meridians_and_parallels(the_basemap, step_degrees = 5.0):
    meridians = np.arange(-180,180, step_degrees)
    parallels = np.arange(-90,90, step_degrees)
    the_basemap.drawparallels(parallels,labels=[0,0,0,0],fontsize=16,linewidth=0.25)
    the_basemap.drawmeridians(meridians,labels=[0,0,0,0],fontsize=16,linewidth=0.25)

if __name__ == "__main__":
    print "Hello World"
