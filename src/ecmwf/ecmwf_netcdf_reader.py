from scipy.spatial.kdtree import KDTree
from util import plot_utils
from util.geo import lat_lon

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

from plot2D.map_parameters import polar_stereographic

import itertools

from data import quebec_route_domain_crcm4


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



        ds = nc.Dataset(data_path)
        self.varValues = ds.variables[varName][:].transpose((0,2,1))
        self.timeValues = ds.variables["time"][:]
        self.timeUnits = ds.variables["time"].units
        self.missing_value = ds.variables[varName].missing_value
        self.varUnits = ds.variables[varName].units
        self.longitudes = ds.variables["longitude"][:]
        self.latitudes = ds.variables["latitude"][:]
        self.ecmwf_data_step = timedelta(hours = int(self.timeValues[1] - self.timeValues[0]))

        ds.close()



        lon_0 = self.longitudes[0] + self.longitudes[-1]
        lon_0 /= 2.0
        llcrnrlon = self.longitudes.min()
        llcrnrlat = self.latitudes.min()
        urcrnrlon = self.longitudes.max()
        urcrnrlat = self.latitudes.max()
        self.basemap = Basemap(#projection="cyl",
                    #lat_0=45, lon_0 = -40,
                    llcrnrlon=llcrnrlon,
                    llcrnrlat=llcrnrlat,
                    urcrnrlon=urcrnrlon,
                    urcrnrlat=urcrnrlat
        )

        [lats2d, lons2d] = np.meshgrid(self.latitudes, self.longitudes)
        print "lon range: ", lons2d.min(), lons2d.max()
        print "lat range: ", lats2d.min(), lats2d.max()
        print "lower left crnr (lon, lat): ", lons2d[0,0], lats2d[0,0]
        print "upper right crnr (lon, lat): ", lons2d[-1,-1], lats2d[-1,-1]
        [x, y, z] = lat_lon.lon_lat_to_cartesian_normalized(lons2d.flatten(), lats2d.flatten())

        grid_points = np.array([x, y, z]).transpose()
        self.kd_tree = KDTree(grid_points)

        pass

    def get_lat_lon_2d(self):
        return np.meshgrid(self.latitudes, self.longitudes)

    def get_data_interpolated_to_points(self, dest_lons = None,
                                        dest_lats = None,
                                        source_lons = None,
                                        source_lats = None,
                                        data = None):

        """
        Designed to interpolate all data  to the AMNO domain
        """
        if None not in [source_lons, source_lats]:
            lons1d = source_lons.flatten()
            lats1d = source_lats.flatten()

            points = lat_lon.lon_lat_to_cartesian_normalized(lons1d, lats1d)
            points = np.array(points).transpose()
            point_tree = KDTree(points)
        else:
            point_tree =  self.kd_tree


        assert source_lons.shape == source_lats.shape == data.shape

        [xi, yi, zi] = lat_lon.lon_lat_to_cartesian_normalized(dest_lons.flatten(), dest_lats.flatten())
        pointsi = np.array([xi, yi, zi]).transpose()
        data1d = data.flatten()
        #distances dimensions = (n_points, n_neigbours)
        [distances, indices] = point_tree.query(pointsi, k = 4)


        weights = 1.0 / distances ** 2
        norm = [np.sum(weights, axis = 1)] * weights.shape[1]
        norm = np.array(norm).transpose()
        weights /= norm

        result = []
        for i in xrange(pointsi.shape[0]):
            w = weights[i, :]
            d = data1d[indices[i, :]]

            result.append(np.sum(w * d))
        return np.array(result).reshape( dest_lons.shape )





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
        #raise Exception('Quebec feature is not found')

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


        self.quebec_mask = np.zeros((len(self.longitudes), len(self.latitudes)))

        for i, lon in enumerate(self.longitudes):
            for j, lat in enumerate(self.latitudes):
                p = ogr.CreateGeometryFromWkt('POINT('+str(lon if lon <= 180 else lon - 360) + ' ' + str(lat) + ')')
                self.quebec_mask[i, j] = int( thePolygon.Contains(p))




        pass

    def get_interpolation_weights(self, source_lons = None, source_lats = None,
                                       dest_lons = None, dest_lats = None, nneighbours = 4
                                 ):
        """
        get the interpolation array M (nx*ny, nneighbors) and negibor indices G:

        DEST_FIELD = M * SOURCE_FIELD.flatten()[G]
        """

        source_lons_1d, source_lats_1d = source_lons.flatten(), source_lats.flatten()
        [x,y,z] = lat_lon.lon_lat_to_cartesian_normalized(source_lons_1d, source_lats_1d)
        kdtree = KDTree(zip(x, y, z))

        [xi, yi, zi] = lat_lon.lon_lat_to_cartesian_normalized(dest_lons.flatten(), dest_lats.flatten())

        [distances, indices] = kdtree.query( zip( xi, yi, zi ) , k = nneighbours )




        if len(distances.shape) == 2:
            weights = 1.0 / distances ** 2
            norm = weights.sum(axis = 1)
            norm = np.array( [norm] * nneighbours ).transpose()
            weights /= norm
        else:
            weights = np.ones(distances.shape)


        return weights, indices


    def interpolate_to_amno_ps_all_data_and_save(self, result_file = "rof_amno_ecmwf.nc"):
        """
        Interpolates the data read from the current ecmwf netcdf file to the
        North polar stereographic grid
        and save the result to file
        """
        dest_lons, dest_lats = polar_stereographic.lons, polar_stereographic.lats
        [source_lats, source_lons] = self.get_lat_lon_2d()

        [weights, indices] = self.get_interpolation_weights(source_lons=source_lons, source_lats=source_lats,
                                        dest_lons=dest_lons, dest_lats=dest_lats, nneighbours=1
                            )


        ds = nc.Dataset(result_file, mode="w", format='NETCDF3_CLASSIC')
        ds.createDimension("time", size = self.varValues.shape[0])
        ds.createDimension("longitude", size=dest_lons.shape[0])
        ds.createDimension("latitude", size=dest_lons.shape[1])

        varDest = ds.createVariable("runoff", "f4", ("time", "longitude", "latitude") )
        lonVar = ds.createVariable("longitude", "f4", ( "longitude", "latitude"))
        latVar = ds.createVariable("latitude", "f4", ( "longitude", "latitude" ))

        lonVar[:, :] = polar_stereographic.lons
        lonVar.units = "degrees east"
        latVar[:, :] = polar_stereographic.lats
        latVar.units = "degrees north"

        for t in xrange(self.varValues.shape[0]):
            data_t = self.varValues[t, :, :]
            #w = weights.copy()
            #w[data_t == self.missing_value] = 0 #this is in order to not interpolate the missing values
            data_dest = weights * data_t.flatten()[indices]
            if len(data_dest.shape) == 2:
                data_dest = data_dest.sum(axis = 1)
            varDest[t, :, :] = np.reshape(data_dest, dest_lons.shape)

        # units conversion
        varDest.units = "kg / (m**2 * s)"
        water_density = 1000.0
        varDest[:] = varDest[:] * water_density / float( self.ecmwf_data_step.seconds )

        #save time data
        timeVar = ds.createVariable("time", "i4", ("time"))
        timeVar[:] = self.timeValues[:]
        timeVar.units = self.timeUnits

        plt.figure()
        b = polar_stereographic.basemap

        [x, y] = b(dest_lons, dest_lats)

        b.llcrnrx = x.min()
        b.llcrnry = y.min()
        b.urcrnrx = x.max()
        b.urcrnry = y.max()



        b.pcolormesh(x, y, varDest[:].mean(axis = 0))
        b.drawcoastlines()
        b.drawmeridians(np.arange(-180, 180, 20), labels = [1,0,0,1])
        b.drawparallels(np.arange(-90, 90, 20))

        b.drawmapboundary()
        plt.colorbar()
        plt.savefig("rof_mean.png")
        ds.close()



        pass


    def calculate_monthly_normals(self):
        """
        calculates
        """

        ds = nc.Dataset(self._data_path)
        evap_var = ds.variables[self.varName]
        self.missing_value = -evap_var.missing_value
        evap = evap_var[:,:,:]
        #minus because the downward fluxes are positive in the ECMWF model
        evap = -np.transpose(evap, (0,2,1)) #make the order t,lon,lat

        self.mean_per_year = np.sum(evap, axis = 0) / 20.0

        time_var = ds.variables['time']
        time_format = '%Y-%m-%d %H:%M:%S.0'
        units = time_var.units.split('since')[-1].strip()
        self.reference_date = datetime.strptime(units, time_format)
        print self.reference_date

        hours_since_reference = time_var[:]
        #get datetime objects
        time_objects = map( lambda x: timedelta(hours = int(x)) + self.reference_date,
            hours_since_reference )


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
        print 'end date: ', end_date


        nyears = end_date.year - start_date.year + 1
        print "nyears = ", nyears
        for i in xrange(12):
            x = monthly_fields[i] / float(nyears)
            self.monthly_normals.append(x)




        lons = ds.variables['longitude'][:]
        self.latitudes = ds.variables['latitude'][:]
        #self.latitudes = self.latitudes[::-1]


        #convert to -180, 180
        #lons[lons >= 180] -= 360
        self.longitudes = lons
        pass

    def calculate_average_for_domain_and_plot(self):
        condition = (self.quebec_mask == 1)
        averages_for_domain = []
        max_for_domain = []
        min_for_domain = []

        [lats2d, lons2d] = self.get_lat_lon_2d()

        [x, y] = self.basemap(lons2d, lats2d)

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

        #plot annual mean field
        plt.figure()
        to_plot =  np.sum( self.monthly_normals, axis=0 )

        #to_plot = np.ma.masked_where(  )
        self.basemap.pcolormesh(x,y, to_plot )#, vmin = 0.0, vmax = 0.24)
        plt.colorbar()
        self.basemap.drawcoastlines()
        plt.savefig("ecmwf_evap_annual_mean_field.png")

#        plt.show()
        plt.figure()
        plt.plot(averages_for_domain, lw = 3, label = "Domain mean")
        plt.plot(max_for_domain, lw = 3, label = "Domain max")
        plt.plot(min_for_domain, lw = 3, label = "Domain min")
        plt.legend()

        plt.ylabel('m / month')
        plt.title('%s normals averaged over Quebec region' % self.varName)
        plt.xticks(xrange(12), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        plt.xlabel('yearly value is %f ' % sum(averages_for_domain))
        plt.savefig('{0}_qc.png'.format(self.varName))


def calculate_runoff_staff():
    start_date = None #datetime(1980,1,1)
    end_date = None #datetime(1999, 12, 31)


    er = EcmwfReader( #data_path="data/era_interim/evap-1980-1999_global.nc",
        data_path="data/era_interim/runoff-1980-1999.nc",
        start_date = start_date, end_date = end_date, varName="ro")

    er.interpolate_to_amno_ps_all_data_and_save()

    pass

def calculate_evap_staff():
    start_date = None #datetime(1980,1,1)
    end_date = None #datetime(1999, 12, 31)


    er = EcmwfReader( #data_path="data/era_interim/evap-1980-1999_global.nc",
        data_path="data/era_interim/evap_day_night-1980-1999.nc",
        start_date = start_date, end_date = end_date)
    er.calculate_monthly_normals()
    print 'calculated monthly means'
    er.generate_quebec_mask()
    print 'generated quebec mask'


    er.calculate_average_for_domain_and_plot()

    data = np.sum( er.monthly_normals, axis = 0 )


    [era_lats2d, era_lons2d] = er.get_lat_lon_2d()

    condition = ( er.quebec_mask <= 1 )
    data_1d = data[condition]
    era_lons1d = era_lons2d[condition]
    era_lats1d = era_lats2d[condition]

    amno_data = er.get_data_interpolated_to_points(data = data_1d,
        dest_lons=polar_stereographic.lons,
        dest_lats=polar_stereographic.lats,
        source_lons= era_lons1d, source_lats=era_lats1d
    )

    domain_mask = quebec_route_domain_crcm4.get_domain_mask()

    plt.figure()

    basemap = polar_stereographic.basemap
    print polar_stereographic.lons.min(), polar_stereographic.lons.max()
    [x, y] = basemap( polar_stereographic.lons, polar_stereographic.lats )

    amno_data = np.ma.masked_where(domain_mask != 1, amno_data)
    basemap.pcolormesh(x, y, amno_data)
    basemap.drawcoastlines()

    plt.colorbar()
    r = plot_utils.get_ranges(x[domain_mask == 1], y[domain_mask == 1])
    plt.xlim(r[:2])
    plt.ylim(r[2:])

    #basemap.ax.set_xlim(r[:2])
    #basemap.ax.set_ylim(r[2:])

    basemap.drawmeridians(np.arange(-180,180,10))



    basemap.drawmeridians(np.arange(-180, 180, 10), labels = [0,0,0,1])
    basemap.drawparallels(np.arange(-90, 90, 20), labels = [1,0,0,0])
    plt.savefig("evap_interp.png")

    #sel_data = amno_data[domain_mask == 1]
    print "mean evap: %f " % (np.ma.mean(amno_data))
    #print "mean evap %f " % (np.sum(er.monthly_normals, axis = 0).mean())
    print "Hello World"



if __name__ == "__main__":
    application_properties.set_current_directory()
    calculate_runoff_staff()
    #calculate_evap_staff()