__author__="huziy"
__date__ ="$Aug 30, 2011 10:00:02 PM$"

import matplotlib.pyplot as plt

import numpy as np

def test():
    x = np.random.rand(10, 10)
    plt.pcolormesh(x)
    
    the_is = []
    the_js = []
    x_flat = []
    nx, ny = x.shape
    for i in xrange(nx):
        for j in xrange(ny):
            the_is.append(i)
            the_js.append(j)
            x_flat.append(x[i, j])
    plt.scatter(the_is, the_js, c = 'k', s = 10)
    plt.show()
    print x

if __name__ == "__main__":
    test()
    print "Hello World"
