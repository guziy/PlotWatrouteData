import codecs
import re
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib as mpl
import netCDF4

__author__="huziy"
__date__ ="$Aug 4, 2011 5:06:41 PM$"

from util.geo.GeoPoint import GeoPoint
import os
import numpy as np

import application_properties
from datetime import datetime
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from netCDF4 import date2num, num2date
from matplotlib.ticker import LogLocator
from matplotlib import colors

class SweHolder:
    """
    point (1,1) - is a lower left point of the grid
    """
    def __init__(self, path = 'data/swe_ross_brown/derksen'):
        self.root_path = path
        self.step = 0.25
        self.lowerLeft = GeoPoint(longitude = -160.0, latitude = 0.0)
        self.nEastWest = 480
        self.nNorthSouth = 360
        self.file_name_format = '%Y%m%d%H'

        self.all_dates = []


        self.pathToNetCDF = 'data/swe_ross_brown/swe.nc'
        self._initDates()
        self.lons2d, self.lats2d = self.get2DLonsAndLats()


    def get2DLonsAndLats(self):
        """
        point (i = 0, j = 0) is at the lower left corner of the domain
        """
        lon0 = self.lowerLeft.longitude
        lat0 = self.lowerLeft.latitude


        lons = np.arange(lon0, lon0 + self.nEastWest * self.step, self.step)
        lats = np.arange(lat0, lat0 + self.nNorthSouth * self.step, self.step)

        lats, lons = np.meshgrid(lats, lons)
        return lons, lats


    def get_mean_for_months(self, months = None):
        if not months: months = []
        pass


    def _initDates(self):
        self.all_dates = []
        for fName in os.listdir(self.root_path):
            self.all_dates.append( datetime.strptime( fName.split(".")[0], self.file_name_format))
        self.all_dates.sort()


    def getStartDate(self):
        return min(self.all_dates)

    def getEndDate(self):
        return max(self.all_dates)

    def getTemporalMeanDataFromNetCDFforPoints(self, geopointList=None, startDate=None, endDate=None, months=None):
        """
        months - months of year over which the average is taken
        """
        if not months: months = []
        if not geopointList: geopointList = []
        i_indices = []
        j_indices = []
        for point in geopointList:
            dlon = point.longitude - self.lowerLeft.longitude
            dlat = point.latitude - self.lowerLeft.latitude
            i = int(dlon / self.step)
            j = int(dlat / self.step)
            i_indices.append(i)
            j_indices.append(j)

        ds = Dataset(self.pathToNetCDF)

        times = ds.variables['time']
        times = num2date(times[:], units = times.units)

        data = ds.variables['SWE']
        query = (times >= startDate) & (times <= endDate)
        for i in xrange(len(times)):
            if not query[i]: continue
            if times[i].month not in months:
                query[i] = False
        return np.mean(data[query, i_indices, j_indices], axis = 0)

    def _get_na_0_25_mask(self, path = "data/swe_ross_brown/na_lmask_edit.txt"):
        """
        reads in 0 and 1 mask of na, source : Ross Brown
        """
        f = open(path)
        lines = f.readlines()
        f.close()
        mask = map(lambda x: map(int, re.findall("\d", x)), lines )
        return np.array(mask).transpose()[:,::-1]


    def getSpatialIntegralFromNetcdfForPoints(self, geopointList=None, startDate=None, endDate=None, months=None):
        """
        months - months of year over which the average is taken
        """
        if not months: months = []
        if not geopointList: geopointList = []
        i_indices = []
        j_indices = []
        for point in geopointList:
            dlon = point.longitude - self.lowerLeft.longitude
            dlat = point.latitude - self.lowerLeft.latitude
            i = int(dlon / self.step)
            j = int(dlat / self.step)
            i_indices.append(i)
            j_indices.append(j)

        ds = Dataset(self.pathToNetCDF)

        times = ds.variables['time']
        times = num2date(times[:], units = times.units)

        data = ds.variables['SWE']
        if not len(months):
            query = (times >= startDate) & (times <= endDate)
            return times[query][:], np.sum(data[query, i_indices, j_indices], axis = 1)
        else:
            query = (times >= startDate) & (times <= endDate)
            for i in xrange(len(times)):
                if not query[i]: continue
                if times[i].month not in months:
                    query[i] = False
            return times[query][:], np.sum(data[query, i_indices, j_indices], axis = 1)


    def getSpatialMeanDataFromNetCDFforPoints(self, geopointList=None, startDate=None, endDate=None, months=None):
        """
        months - months of year over which the average is taken
        """
        if not months: months = []
        if not geopointList: geopointList = []
        i_indices = []
        j_indices = []
        for point in geopointList:
            dlon = point.longitude - self.lowerLeft.longitude
            dlat = point.latitude - self.lowerLeft.latitude
            i = int(dlon / self.step)
            j = int(dlat / self.step)
            i_indices.append(i)
            j_indices.append(j)

        ds = Dataset(self.pathToNetCDF)

        times = ds.variables['time']
        times = num2date(times[:], units = times.units)

        data = ds.variables['SWE']
        if not len(months):
            query = (times >= startDate) & (times <= endDate)
            return times[query][:], np.mean(data[query, i_indices, j_indices], axis = 1) 
        else:
            query = (times >= startDate) & (times <= endDate)
            for i in xrange(len(times)):
                if not query[i]: continue
                if times[i].month not in months:
                    query[i] = False
            return times[query][:], np.mean(data[query, i_indices, j_indices], axis = 1)



    def _readFromTxtFile(self, path):
        """
        return data in form of matrix (nx * ny) from path
        """
        data = []
        f = codecs.open(path,mode="r", encoding="utf8")
        lines = f.readlines()
        #lines = map(lambda x: x.rstrip(), lines)
        for line in lines:
            #data.extend(map(float, re.findall("\d{1,4}", line)))
            data.extend(map(float, line.split()))
        data = np.array(data)
        data = np.reshape(data, (self.nEastWest, self.nNorthSouth), order="F")
        data = data[:,::-1]
        f.close()
        return data

    def convertAllDataToNetcdf(self, pathToNetCDF = 'data/swe_ross_brown/swe.nc'):

        ds = Dataset(pathToNetCDF, mode = 'w', format = 'NETCDF4_CLASSIC')
        

        ds.createDimension('time', len(self.all_dates))
        ds.createDimension('lon', self.lons2d.shape[0])
        ds.createDimension('lat', self.lons2d.shape[1])

        lonVariable = ds.createVariable('longitude', 'f4', ('lon', 'lat'))
        latVariable = ds.createVariable('latitude', 'f4', ('lon', 'lat'))

        sweVariable = ds.createVariable('SWE', 'f4', ('time', 'lon', 'lat'))
        sweVariable.units = 'mm of equivalent water'

        timeVariable = ds.createVariable('time', 'i4', ('time'))
        ncSinceFormat = '%Y-%m-%d %H:%M:%S'
        timeVariable.units = ' hours since ' + self.getStartDate().strftime(ncSinceFormat)


        lonVariable[:] = self.lons2d[:, :]
        latVariable[:] = self.lats2d[:, :]


        nDates = len(self.all_dates)
        for i, theDate in enumerate(self.all_dates):
            filePath = os.path.join(self.root_path, theDate.strftime(self.file_name_format) + ".txt")
            data = self._readFromTxtFile(filePath)
            sweVariable[i, :, :] = data[:, :]
            print ' {0} / {1} '.format(i, nDates)

        timeVariable[:] = date2num( self.all_dates, units = timeVariable.units)
        ds.close()



    def getMeanDataForPoints(self, geopointList=None, startDate=None, endDate=None):
        """
            returns sorted list of dates and corresponding list of values
        """
        if not geopointList: geopointList = []
        i_indices = []
        j_indices = []
        for point in geopointList:
            dlon = point.longitude - self.lowerLeft.longitude
            dlat = point.latitude - self.lowerLeft.latitude
            i = int(dlon / self.step)
            j = int(dlat / self.step)
            i_indices.append(i)
            j_indices.append(j)


        dates = []
        result = {}
        for fileName in os.listdir(self.root_path):
            d = datetime.strptime(fileName, self.file_name_format)
            if startDate is not None and endDate is not None:
                if d < startDate or d > endDate:
                    continue

            filePath = os.path.join(self.root_path, fileName)


            data = self._readFromTxtFile(filePath)

            dates.append(d)
            print fileName
            result[d] = np.mean(data[i_indices, j_indices])
        dates.sort()
        return dates, [result[d] for d in dates]


        pass

    def getDataClosestTo(self, geopoint):
        """
        Returns a map {time => value(mm)}
        """
        dlon = geopoint.longitude - self.lowerLeft.longitude
        dlat = geopoint.latitude - self.lowerLeft.latitude
        i = int(dlon / self.step)
        j = int(dlat / self.step)
        print i, j

        if i < 0 or j < 0 or i >= self.nEastWest or j >= self.nNorthSouth:
            raise Exception("indices i = {0}; j = {1} are out of the domain bounds".format(i, j))

        result = {}
        

        for fileName in os.listdir(self.root_path):
            filePath = os.path.join(self.root_path, fileName)
            data = self._readFromTxtFile(filePath)
            d = datetime.strptime(fileName, self.file_name_format)
            result[d] = data
            return

        pass

    def plot_mean(self, year = None, month = None):

        data = []
        if year is not None:
            start = "%d%02d" % (year, month)
            start_index = 0
        else:
            start = "%02d" % month
            start_index = -10

        for the_fname in os.listdir(self.root_path):
            if the_fname[start_index:].startswith(start):
                print the_fname[start_index:]
                x = self._readFromTxtFile(os.path.join(self.root_path, the_fname))
                data.append(x)
        the_mean = np.mean(data, axis = 0)


        plt.figure()
        b = Basemap(projection="cyl", llcrnrlon=self.lons2d[0,0], llcrnrlat=self.lats2d[0,0],
            urcrnrlon=self.lons2d[-1,-1], urcrnrlat=self.lats2d[-1, -1]
        )
        x, y = b(self.lons2d, self.lats2d)
        levels = [10,] + range(20, 120, 20) + [150,200, 300,500,1000]
        cmap = mpl.cm.get_cmap(name="jet_r", lut = len(levels))
        norm = colors.BoundaryNorm(levels, cmap.N)

        print np.ma.max(the_mean)

        the_mask = self._get_na_0_25_mask()
        the_mean = np.ma.masked_where(the_mask == 0, the_mean)

        img = b.contourf(x, y, the_mean, levels = levels, cmap = cmap, norm = norm)
        plt.colorbar(img,ticks = levels, boundaries = levels)
        b.drawcoastlines()
        plt.title("mean over {0}-{1}".format(month, year))
        plt.savefig("swe_mean_{0}_{1}.png".format(month, year))

        pass


    def plot(self, path = ''):
        data = self._readFromTxtFile(path)
        plt.figure()
        b = Basemap(projection="cyl", llcrnrlon=self.lons2d[0,0], llcrnrlat=self.lats2d[0,0],
            urcrnrlon=self.lons2d[-1,-1], urcrnrlat=self.lats2d[-1, -1]
        )
        x, y = b(self.lons2d, self.lats2d)
        b.contourf(x, y, data)
        plt.colorbar()
        b.drawcoastlines()
        plt.title(os.path.basename(path))
        plt.savefig(os.path.basename(path) + ".png")

    pass

def test():
    application_properties.set_current_directory()
    swe = SweHolder(path="data/swe_ross_brown/B2003_daily_swe")
    print netCDF4.__version__
    swe.convertAllDataToNetcdf()

#    swe.plot(path = "data/swe_ross_brown/B2003_daily_swe/1985010112.txt")
#    swe.plot(path = 'data/swe_ross_brown/derksen/1985010112')
#    swe.plot(path = 'data/swe_ross_brown/derksen/1985080112')
#    plt.show()

    #plt.savefig("swe_1985042312.png")
    #swe.plot_mean(year=None,month = 3)

if __name__ == "__main__":
    application_properties.set_current_directory()
    test()
    print "Hello World"
