__author__="huziy"
__date__ ="$Sep 1, 2011 10:26:36 AM$"

from mpl_toolkits.basemap import Basemap
import application_properties
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import netCDF4 as nc
import numpy as np

from osgeo import ogr
from osgeo import gdal

import itertools

class EcmwfReader(object):
    def __init__(self, data_path = 'data/era_interim/evap1980-2010.nc',
                 varName = 'e',
                 start_date = None, end_date = None):

        self.start_date = start_date
        self.end_date = end_date
        self._data_path = data_path
        self.monthly_normals = []
        self.longitudes = None
        self.latitudes = None
        self.reference_date = None
        self.varName = varName

        self.quebec_mask = None # 1 for the points in Quebec 0 for the points outside

        self.missing_value = None

        self._calculate_monthly_normals()
        print 'calculated monthly means'
        self.generate_quebec_mask()
        print 'generated quebec mask'


        lon_0 = self.longitudes[0] + self.longitudes[-1]
        lon_0 /= 2.0
        llcrnrlon = self.longitudes[0]
        llcrnrlat = self.latitudes.min()
        urcrnrlon = self.longitudes[-1]
        urcrnrlat = self.latitudes.max()
        self.basemap = Basemap(projection="lcc",
                    lat_0=45, lon_0 = lon_0,
                    llcrnrlon=llcrnrlon,
                    llcrnrlat=llcrnrlat,
                    urcrnrlon=urcrnrlon,
                    urcrnrlat=urcrnrlat

        )
        pass



    def get_qc_polygon_object(self, path_to_shapefile = 'data/shape/canada_provinces_shape/PROVINCE.SHP'):
        """
        we cannot return geometries since they are stored in the
        separate memory from the one of the program, that is why
        returning feature
        """
        print 'inside get_qc_polygon_object'
        gdal.UseExceptions()
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataStore = driver.Open(path_to_shapefile, 0)

        layer = dataStore.GetLayer(0)

        print dataStore.GetLayerCount()
        for feature in layer:
            print feature.GetFieldAsString('NAME').lower()
            if 'quebec' in feature.GetFieldAsString('NAME').lower():
                return feature
        raise Exception('Quebec feature is not found')

    def generate_quebec_mask(self):
        """
        Generates mask of the quebec region using the
        oficial shape for the province borders
        """
        if self.longitudes is None or self.latitudes is None:
            print 'You should define the longitudes and latitudes of the grid beforehand'
            raise Exception("exception occurred")

        thePolygonFeature = self.get_qc_polygon_object()
        thePolygon = thePolygonFeature.GetGeometryRef()


        b = Basemap()
        [x1, y1] = b(self.longitudes, self.latitudes)
        self.quebec_mask = np.zeros((len(self.longitudes), len(self.latitudes)))

        for i, lon in enumerate(self.longitudes):
            for j, lat in enumerate(self.latitudes):
                p = ogr.CreateGeometryFromWkt('POINT('+str(lon) + ' ' + str(lat) + ')')
                self.quebec_mask[i, j] = int( thePolygon.Contains(p) and b.is_land(x1[i], y1[j]))


        pass

    def _calculate_monthly_normals(self):
        """
        calculates
        """
        print nc.__version__
        ds = nc.Dataset(self._data_path)
        evap_var = ds.variables[self.varName]
        self.missing_value = evap_var.missing_value
        evap = evap_var[:,:,:]
        evap = -np.transpose(evap, (0,2,1)) #make the order t,lon,lat

        print evap.shape

        time_var = ds.variables['time']
        time_format = '%Y-%m-%d %H:%M:%S.0'
        units = time_var.units.split('since')[-1].strip()
        self.reference_date = datetime.strptime(units, time_format)
        print self.reference_date

        hours_since_reference = time_var[:]
        #get datetime objects
        time_objects = map( lambda x: timedelta(hours = int(x)) + self.reference_date, hours_since_reference )


        #reject the dates that are less than start_date
        if self.start_date is not None:
            time_objects_select = itertools.ifilter(lambda x: x >= self.start_date, time_objects)
            time_objects_select = list(time_objects_select)
            time_start_index = time_objects.index(time_objects_select[0])
            time_objects = time_objects_select
            evap = evap[time_start_index:,:,:]
            print "time_start_index = ", time_start_index
        print self.start_date



        #reject the dates that are later than end_date
        if self.end_date is not None:
            time_objects_select = itertools.ifilter(lambda x: x <= self.end_date, time_objects)
            time_objects_select = list( time_objects_select )
            time_end_index = time_objects.index(time_objects_select[-1])
            time_objects = time_objects_select
            evap = evap[0:(time_end_index + 1),:,:]
            print "time_end_index = ", time_end_index
        monthly_fields = []


        assert evap.shape[0] == len(time_objects)
        print time_objects[0], time_objects[-1]

        for the_month in xrange(1, 13):
            bool_vector = map( lambda x: x.month == the_month, time_objects)
            indices = np.where( bool_vector )[0]
            monthly_fields.append(np.sum(evap[indices, :, :] , axis=0)) #sum for a given month



        start_date = time_objects[0]
        end_date = time_objects[-1]

        print 'start date: ', start_date
        print 'start date: ', end_date


        nyears = end_date.year - start_date.year + 1
        for i in xrange(12):
            x = monthly_fields[i] / float(nyears)
            self.monthly_normals.append(x)

        lons = ds.variables['longitude'][:]
        self.latitudes = ds.variables['latitude'][:]
        #self.latitudes = self.latitudes[::-1]


        #convert to -180, 180
        lons[lons >= 180] -= 360
        self.longitudes = lons
        pass

    def calculate_average_for_domain_and_plot(self):
        condition = (self.quebec_mask == 1)
        print len(self.quebec_mask[condition])
        averages_for_domain = []
        max_for_domain = []
        min_for_domain = []

        [y, x] = np.meshgrid(self.latitudes,self.longitudes )

        [x, y] = self.basemap(x, y)

        plt.figure()
        self.basemap.pcolormesh(x, y, self.quebec_mask)
        self.basemap.drawcoastlines()
        plt.colorbar()
        plt.savefig("quebec_mask.png")


        for i, field in enumerate(self.monthly_normals):
            plt.figure()
            self.basemap.pcolormesh(x, y, field)
            self.basemap.drawcoastlines()
            plt.colorbar()
            plt.savefig("%d_evap_per_month_field.png" % (i+1))
            selected = field[condition & (field != self.missing_value)]
            averages_for_domain.append(np.mean(selected))
            max_for_domain.append(np.max(selected))
            min_for_domain.append(np.min(selected))
#        plt.show()
        plt.figure()
        plt.plot(averages_for_domain, lw = 3, label = "Domain mean")
        plt.plot(max_for_domain, lw = 3, label = "Domain max")
        plt.plot(min_for_domain, lw = 3, label = "Domain min")
        plt.legend()

        plt.ylabel('m / month')
        plt.title('%s normals integrated over Quebec region' % self.varName)
        plt.xticks(xrange(12), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        plt.xlabel('yearly value is %f ' % sum(averages_for_domain))
        plt.savefig('{0}_qc.png'.format(self.varName))

        
    def get_average_interpolated_to_points(self, lons = None, lats = None):
        """
        Interpolate normals to the points lons, lats, and then take average over
        the points
        returns a list (of length 12) of normals for each month, averaged
        over the points
        """


        pass




if __name__ == "__main__":
    application_properties.set_current_directory()
    start_date = datetime(1980,1,1)
    end_date = datetime(2010, 12, 31)
    er = EcmwfReader(start_date = start_date, end_date = end_date)
    er.calculate_average_for_domain_and_plot()

    print "Hello World"
