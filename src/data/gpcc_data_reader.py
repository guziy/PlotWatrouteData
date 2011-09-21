__author__="huziy"
__date__ ="$Aug 30, 2011 11:40:23 PM$"

from osgeo import ogr
import matplotlib.pyplot as plt
import numpy as np

class GPCCData:
    def __init__(self):
        self.monthlyData = []
        self.yearlyData = []
        self.lons = []
        self.lats = []
        self._parse_data()
        self.resolution = None

        pass

    def _parse_data(self, path = 'data/gpcc/RASTER_DATA/gpcc_precipitation_normals_rr_version_2010_0_50_degree',
                          resolution = 0.5):
        self.resolution = resolution

        f = open(path)
        lines = f.readlines()
        iStart = int(lines[0].split()[0])
        lonStart = -180
        lonEnd = 180
        latStart = 90
        latEnd = -90

        lat = latStart
        lon = lonStart
        for line in lines[iStart:]:
            if lon >= lonEnd:
                lon = lonStart
                lat -= resolution
            fields = line.split()
            self.lons.append(lon)
            self.lats.append(lat)

            self.monthlyData.append(map(float, fields[:12]))
            self.yearlyData.append(float(fields[12]))
            lon += resolution
        print 'lat, lon = ', lat, lon

        #assuming that the domain is rectangular
        self.lowerLeftLon = np.min(self.lons)
        self.lowerLeftLat = np.min(self.lats)
        self.upperRightLon = np.max(self.lons)
        self.upperRightLat = np.max(self.lats)

        self.yearlyData = np.array(self.yearlyData)
        self.monthlyData = np.array(self.monthlyData)

        self.lons = np.array(self.lons)
        self.lats = np.array(self.lats)
        f.close()
        pass

    def _calculate_distances(self, lon, lat):
        return map(lambda x: (lon - x[0]) ** 2 + (lat - x[1]) ** 2, zip(self.lons, self.lats))

    def get_monthly_precipitations_for_point(self, lon, lat):
        """
        select monthly means for the given point,
        used to compare 2 points
        """
        distances = self._calculate_distances(lon, lat)
        closest = distances == np.min(distances)
        return self.monthlyData[closest][0]

    def get_monthly_mean_over_points(self, the_lons, the_lats):
        """
        given the list of longitudes and latitudes, get spatial mean of monthly
        normals over the given points
        """
        prec_for_all = []
        for x, y in zip(the_lons, the_lats):
            prec1 = self.get_monthly_precipitations_for_point(x, y)
            prec_for_all.append(prec1)

        #get mean over the points
        return np.mean(prec_for_all, axis = 0)
        pass

    def get_integral_over_points_of_monthly_means(self, the_lons, the_lats):
        """
        given the list of longitudes and latitudes, get spatial mean of monthly
        normals over the given points
        """
        prec_for_all = []
        for x, y in zip(the_lons, the_lats):
            prec1 = self.get_monthly_precipitations_for_point(x, y)
            prec_for_all.append(prec1)

        #get sum over the points
        return np.sum(prec_for_all, axis = 0)
        pass



    def calculate_yearly_precip_for_region(self, lon_min, lon_max, lat_min, lat_max):
        result = 0
        monthly = [0] * 12
        for lon, lat, value, theMonthsData in zip(self.lons, self.lats, self.yearlyData, self.monthlyData):
            if value < 0:
                continue
            if lon < lon_min or lon > lon_max:
                continue
            if lat < lat_min or lat > lat_max:
                continue

            for i in xrange(12):
                monthly[i] += theMonthsData[i]
            result += value
        print result
        print monthly

    def calculate_yearly_precip_for_shape(self, path = 'data/shape/canada_provinces_shape/PROVINCE.SHP'):
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataStore = driver.Open(path, 0)
        print 'number of layers in the file: ', dataStore.GetLayerCount()

        layer = dataStore.GetLayer(0)
        print 'number of features in the file: ', layer.GetFeatureCount()
        print 'layer name: ', layer.GetName()



        featureDef = layer.GetLayerDefn()

        print 'field names for each feature in the shape file:'
        for i in xrange(featureDef.GetFieldCount()):
            print featureDef.GetFieldDefn(i).GetName()

        print featureDef.GetName()


        #select polygon corresponding to quebec
        feature = layer.GetNextFeature()
        while feature:
            if 'quebec' in feature.GetFieldAsString('NAME').lower():
                break
            feature = layer.GetNextFeature()

        print dir(feature)
        print dir(layer)
        print 'shape extent: ', layer.GetExtent()
        print 'spatial ref: ', layer.GetSpatialRef(), layer.Reference()
        print layer.GetFIDColumn()


        print feature.DumpReadable()

        geom = feature.GetGeometryRef()

        print geom.GetGeometryType() == ogr.wkbPolygon

        result = 0
        monthly = [[] for i in xrange(12)]
        for lon, lat, value, theMonthsData in zip(self.lons, self.lats, self.yearlyData, self.monthlyData):
            if value < 0:
                continue

            p = ogr.CreateGeometryFromWkt('POINT('+str(lon) + ' ' + str(lat) + ')')
            if not geom.Contains(p):
                continue

            for i in xrange(12):
                monthly[i] += [theMonthsData[i]]

            result += value



        #average over points in space
        monthly = np.array(monthly)
        print monthly.shape
        monthly = np.mean(monthly, axis = 1)

        #convert to meters per month
        monthly /= 1000.0
        print result
        print monthly
        self._plot_monthly_series(monthly)

    def _plot_monthly_series(self, series):
        plt.figure()
        plt.plot(xrange(12), series, lw = 3)
        plt.xticks(xrange(12), ['Jan','Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.title('Net monthly precipitation normals for 1980-2010')
        plt.ylabel('m/month')
        plt.xlabel('Annual total is %.2f m' % sum(series))
        plt.savefig('qc_precip_normals.pdf', bbox_inches = 'tight')





import application_properties
if __name__ == "__main__":
    application_properties.set_current_directory()
    gpcc = GPCCData()
#    gpcc.calculate_yearly_precip_for_shape()
    gpcc.get_monthly_precipitations_for_point(-75.8, 47.1)
    gpcc.get_monthly_precipitations_for_point(-78, 47)
    print "Hello World"
