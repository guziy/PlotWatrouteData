# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="huziy"
__date__ ="$24 juil. 2010 16:45:38$"

import numpy as np

def draw_meridians_and_parallels(the_basemap, step_degrees = 5.0):
    meridians = np.arange(-180,180, step_degrees)
    parallels = np.arange(-90,90, step_degrees)
    the_basemap.drawparallels(parallels,labels=[0,0,0,0],fontsize=16,linewidth=0.25)
    the_basemap.drawmeridians(meridians,labels=[0,0,0,0],fontsize=16,linewidth=0.25)


if __name__ == "__main__":
    print "Hello World"
