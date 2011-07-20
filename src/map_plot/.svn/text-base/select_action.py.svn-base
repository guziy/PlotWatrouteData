__author__="huziy"
__date__ ="$17 nov. 2010 11:50:55$"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

class HighlightSelected(lines.VertexSelector):
    def __init__(self, line, fmt='ro', **kwargs):
        lines.VertexSelector.__init__(self, line)
        self.markers, = self.axes.plot([], [], fmt, **kwargs)

    def process_selected(self, ind, xs, ys):
        self.markers.set_data(xs, ys)
        self.canvas.draw()


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = np.random.rand(2, 30)
    line, = ax.plot(x, y, 'bs-', picker=5)

    selector = HighlightSelected(line)
    plt.show()


if __name__ == "__main__":
    main()
    print "Hello World"
