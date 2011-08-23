__author__="huziy"
__date__ ="$Aug 17, 2011 3:49:28 PM$"


def apply_plot_params(font_size = 20, width_pt = 1000):
    import pylab
    import math
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (math.sqrt(5.0) - 1.0) / 2.0       # Aesthetic ratio
    fig_width = width_pt * inches_per_pt          # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width,  fig_height]

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



if __name__ == "__main__":
    print "Hello World"
