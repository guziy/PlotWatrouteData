__author__="huziy"
__date__ ="$20 dec. 2010 19:59:26$"


from read_shape_file import *
def plot_basin_boundaries_from_shape(basemap, plotter = None, linewidth = 2):
    ax = plotter.gca()
    for poly in get_features_from_shape(basemap, linewidth = linewidth):
        ax.add_patch(poly)
    pass



if __name__ == "__main__":
    print "Hello World"
