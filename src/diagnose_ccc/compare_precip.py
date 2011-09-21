__author__ = 'huziy'


import compare_swe
from data.cru_data_reader import CruReader

from plot2D.map_parameters import polar_stereographic

import matplotlib.pyplot as plt
import numpy as np


def compare_precip(path_to_ccc = 'data/ccc_data/aex/aex_p1pcp', mask = None, label = ''):
    """
    compare total precip from gpcc and from CRCM4
    """

    ccc_data = compare_swe.getMonthlyNormalsAveragedOverMask(path_to_ccc = path_to_ccc, mask = mask)


    lons = polar_stereographic.lons[mask == 1]
    lats = polar_stereographic.lats[mask == 1]
    cru_reader = CruReader()
    cru_data = cru_reader.get_monthly_normals_integrated_over(lons, lats)

    cru_reader.close_data_connection()


    plt.figure(figsize = (10,5))  #figure width and height in inches
    plt.title("Total precipitation")
    plt.ylabel("mm")
    plt.plot(ccc_data, label = "ccc", lw = 3)
    plt.plot(np.array(cru_data) , label = "cru", lw = 3)
    plt.legend()
    months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    plt.xticks(xrange(12), months)
    plt.savefig("tp_{0}.pdf".format(label))

    print "ccc:"
    print ccc_data
    print "cru:"
    print cru_data

