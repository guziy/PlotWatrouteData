# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="huziy"
__date__ ="$24 juil. 2010 16:45:38$"

import numpy as np



def zoom_to_qc(plotter):
    ymin, ymax = plotter.ylim()
    plotter.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)

    xmin, xmax = plotter.xlim()
    plotter.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)



if __name__ == "__main__":
    print "Hello World"
