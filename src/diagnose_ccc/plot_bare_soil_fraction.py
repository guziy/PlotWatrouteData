from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap, maskoceans

__author__ = 'huziy'

import numpy as np
from ccc.ccc import champ_ccc

from plot2D.map_parameters import polar_stereographic
import matplotlib.pyplot as plt


def main():
    basemap = Basemap(projection = 'npstere', #area_thresh = 10000,
                        lat_ts = 60, lat_0 = 60, lon_0 = -115,
                        boundinglat=10

                        )

    x = polar_stereographic.lons
    y = polar_stereographic.lats

    x, y = basemap(x, y)

    path = "/home/huziy/skynet3_exec1/crcm4_data/AN"
    ccc_obj = champ_ccc(fichier=path)
    records = ccc_obj.charge_champs()
    fields_of_interest = []
    for rec in records:
        name = rec["ibuf"]["ibuf3"]
        level = rec["ibuf"]["ibuf4"]
        if name == "FCAN" and level < 5:
            print rec["ibuf"]
            field = rec["field"]
            print field.min(), field.max()
            fields_of_interest.append(field)
            print rec["field"].shape

    fig = plt.figure()
    assert isinstance(fig, Figure)
    #clevels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    pData = 1.0 - np.sum(fields_of_interest, axis = 0)
    pData = pData[10:190,10:182]
    pData[np.abs( pData ) < 0.001] = 0.0
    pData = maskoceans(polar_stereographic.lons, polar_stereographic.lats,pData,inlands=False)
    print pData.min(), pData.max()
    #pData = np.ma.masked_where((pData > 1) | (pData < 0), pData)
    basemap.contourf(x,y,pData)
    basemap.drawcoastlines()
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.show()
    fig.savefig("bare_soil.png")

    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  