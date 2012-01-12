from mpl_toolkits.basemap import Basemap

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt
import application_properties


def main():
    b = Basemap(projection="lcc",
                        lat_0=55, lon_0 = -50,
                        llcrnrlon=-100,
                        llcrnrlat=30,
                        urcrnrlon=0,
                        urcrnrlat=80
            )
    plt.figure()
    b.drawcoastlines()
    b.drawmeridians(np.arange(-90, 90, 10), labels=[0,0,0,1])
    plt.savefig("test_projections.png")

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  