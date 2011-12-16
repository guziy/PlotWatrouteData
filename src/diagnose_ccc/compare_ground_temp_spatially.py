from datetime import datetime
import os
from matplotlib import mpl
from matplotlib.ticker import LinearLocator
from mpl_toolkits.basemap import Basemap
__author__ = 'huziy'


import application_properties
from ccc import ccc
import matplotlib.pyplot as plt
from plot2D.map_parameters import polar_stereographic
import numpy as np


date_format = "%Y%m"

def get_mean(path = "data/ccc_data/aex/aex_p1gt", start_date = None, end_date = None):
    result = None
    count = 1.0
    for fName in os.listdir(path):
        the_date = fName.split("_")[-1].split(".")[0]
        the_date = datetime.strptime(the_date, date_format)

        if start_date is not None and end_date is not None:
            if the_date < start_date or the_date > end_date:
                continue

        fPath = os.path.join(path, fName)
        cccObj = ccc.champ_ccc( fichier = fPath)
        fields = cccObj.charge_champs()
        #calculate mean
        for the_record in fields:
            if result is None:
                result = the_record["field"]
            else:
                count += 1.0
                result = ( the_record["field"] +  result * (count - 1) ) / count
    return result


def plot_result(field = None):
    if field is None:
        return


    basemap = Basemap(projection = 'npstere', #area_thresh = 10000,
                        lat_ts = 60, lat_0 = 60, lon_0 = -115,
                        boundinglat=10
                        )

    x = polar_stereographic.lons
    y = polar_stereographic.lats

    x, y = basemap(x, y)


    d = max(np.absolute(field.max()), np.absolute(field.min()))
    basemap.pcolormesh(x, y, field, cmap = mpl.cm.get_cmap('jet', 10),
                       vmax = d, vmin = -d)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.colorbar(ticks = LinearLocator(numticks = 11), format = "%.1f")
    basemap.drawcoastlines()
    #basemap.drawrivers()
    plt.savefig("gt_aex_minus_aey.png")

    pass


def main():
    start_date = datetime(1980, 1, 1)
    end_date = datetime(1999, 12, 31)
    mean_aet = get_mean(path="data/ccc_data/aey/aey_p1gt",
                        start_date = start_date, end_date = end_date)
    mean_aex = get_mean(start_date = start_date, end_date = end_date)

    plot_result(field=mean_aex - mean_aet)
    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    main()