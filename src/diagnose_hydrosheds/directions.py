__author__="huziy"
__date__ ="$Jul 23, 2011 12:42:17 PM$"


from netCDF4 import Dataset
import application_properties

application_properties.set_current_directory()

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt


class Plotter:

    def __init__(self, path = ''):
        self.dataset = Dataset(path)

        self.lons = self.dataset.variables['lon'][:]
        self.lats = self.dataset.variables['lat'][:]

        self.lons[self.lons > 180] -= 360

        lon_min, lon_max = np.min(self.lons), np.max(self.lons)
        lat_min, lat_max = np.min(self.lats), np.max(self.lats)


        self.basemap = Basemap(llcrnrlon = lon_min, llcrnrlat = lat_min,
                               urcrnrlon = lon_max, urcrnrlat = lat_max,
                               resolution = 'i'
                                )
        self.lons, self.lats = self.basemap(self.lons, self.lats)



    def plot_accumulation_area(self):
        data = self.dataset.variables['accumulation_area'][:]
        slope = self.dataset.variables['slope'][:]
        flow_direction_index_0 = self.dataset.variables['flow_direction_index0'][:]

        print data.shape

        plt.figure()
        
        data = np.ma.masked_where(slope < 0, data)
        self.basemap.pcolormesh(self.lons, self.lats, data)
        self.basemap.drawcoastlines()
        plt.title('accumulation_area')
        plt.colorbar()
        plt.savefig("acc_area.png")


        plt.figure()
        data = np.ma.masked_where(slope < 0, data)
        self.basemap.pcolormesh(self.lons, self.lats, np.ma.log(data))
        self.basemap.drawcoastlines()
        plt.title('accumulation_area')
        plt.colorbar()



        plt.figure()
        data = np.ma.masked_where(slope < 0, flow_direction_index_0)
        print 'min, max: %f, %f -- for index0' % (np.min(data), np.max(data))
        self.basemap.pcolormesh(self.lons, self.lats, data)
        self.basemap.drawcoastlines()
        plt.title('direction index 0')
        plt.colorbar()

        plt.figure()
        data = np.ma.masked_where(slope < 0, self.dataset.variables['flow_direction_index1'][:])
        print 'min, max: %f, %f -- for index1' % (np.min(data), np.max(data))
        self.basemap.pcolormesh(self.lons, self.lats, data)
        self.basemap.drawcoastlines()
        plt.title('direction index 1')
        plt.colorbar()

        plt.figure()
        channelLength = self.dataset.variables['channel_length'][:]
        indices = np.where((slope >= 0) & (channelLength < 0))
        print channelLength[indices]
        print slope[indices]

        data = np.ma.masked_where(channelLength < 0, channelLength )
        self.basemap.pcolormesh(self.lons, self.lats, data)
        self.basemap.drawcoastlines()
        plt.title('channel length')
        plt.colorbar()


        pass

if __name__ == "__main__":
    Plotter(path="data/hydrosheds/directions_africa_dx0.44deg.nc").plot_accumulation_area()
    #Plotter(path = 'data/hydrosheds/directions_na_dx0.5.nc').plot_accumulation_area()
    #Plotter(path = 'data/hydrosheds/directions_qc_amno.nc').plot_accumulation_area()
    plt.show()

    print "Hello World"
