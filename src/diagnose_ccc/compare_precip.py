__author__ = 'huziy'



from data.cru_data_reader import CruReader

from plot2D.map_parameters import polar_stereographic

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from util import plot_utils

figure = None
cru_reader = None

def format_yaxis(x, pos = None):
    if not pos % 2:
        return ""
    return "{0}".format(x)


def compare_precip(path_to_ccc = 'data/ccc_data/aex/aex_p1pcp', mask = None, label = '',
                   subplot_count = -1, start_date = None, end_date = None, subplot_total = 10):
    """
    compare total precip from gpcc and from CRCM4
    """


    global figure, cru_reader

    import compare_swe
    ccc_data = compare_swe.getMonthlyNormalsAveragedOverMask(path_to_ccc = path_to_ccc, mask = mask,
                                                             startDate = start_date, endDate = end_date)


    lons = polar_stereographic.lons[mask == 1]
    lats = polar_stereographic.lats[mask == 1]

    if cru_reader is None:
        cru_reader = CruReader()

    cru_data = cru_reader.get_monthly_normals_integrated_over(lons, lats,
                                                              start_date=start_date,
                                                              end_date = end_date)
#


    if subplot_count == 1:
        plot_utils.apply_plot_params(font_size=25, width_pt=1200, aspect_ratio=2)
        figure = plt.figure()
        figure.subplots_adjust(hspace = 0.9, wspace = 0.4, top = 0.9)


    if figure is None:
        print "figure object was not created"
        return

    ax = figure.add_subplot(5, 2, subplot_count)
    ax.set_title("Upstream of {0}".format(label))

    #tmp: data stub
#    ccc_data = np.ones((12,)) * 5000
#    cru_data = np.ones((12,)) * 7000


    line1 = ax.plot(ccc_data, label = "CRCM4", lw = 3, color = "b")
    line2 = ax.plot(np.array(cru_data) , label = "CRU", lw = 3, color = "r")
    months = ('', 'Feb', '', 'Apr', '', 'Jun', '', 'Aug', '', 'Oct', '', 'Dec')
    ax.set_xticks(xrange(12))
    ax.set_xticklabels(months)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_yaxis))

    if subplot_count == subplot_total:
        lines = (line1, line2)
        labels = ("CRCM4", "CRU")
        figure.legend(lines, labels, 'upper center')
#        figure.text(0.05, 0.5, 'TOTAL PRECIPITATION (${\\rm mm / month}$)',
#                      rotation=90,
#                      ha = 'center', va = 'center'
#                      )
        figure.savefig("tp.pdf")
        cru_reader.close_data_connection() #close the file


