import ds
import os.path

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


class SweHolder:
    """
    point (1,1) - is a lower left point of the grid
    """
    def __init__(self):
        self.root_path = 'data/swe_ross_brown/derksen'
        self.step = 0.25
        self.lowerLeft = GeoPoint(longitude = -160.0, latitude = 0.0)
        self.nEastWest = 480
        self.nNorthSouth = 360
        self.file_name_format = '%Y%m%d%H'

        self.all_dates = []
        self._initDates()
        self.conversionCoef = 0.1 #to transorm to mm of water

        self.pathToNetCDF = 'data/swe_ross_brown/swe.nc'


    def get2DLonsAndLats(self):
        '''
        point (i = 0, j = 0) is at the lower left corner of the domain
        '''
        lon0 = self.lowerLeft.longitude
        lat0 = self.lowerLeft.latitude

        lons = np.zeros((self.nEastWest, self.nNorthSouth))
        lats = np.zeros((self.nEastWest, self.nNorthSouth))
        for i in xrange(self.nEastWest):
            for j in xrange(self.nNorthSouth):
                lons[i, j] = lon0 + i * self.step
                lats[i, j] = lat0 + j * self.step
        return lons, lats


    def get_mean_for_months(self, months = None):
        if not months: months = []
        pass


    def _initDates(self):
        for fName in os.listdir(self.root_path):
            self.all_dates.append(datetime.strptime(fName, self.file_name_format))
        self.all_dates.sort()

    def getStartDate(self):
        return self.all_dates[0]

    def getEndDate(self):
        return self.all_dates[-1]

    def getTemporalMeanDataFromNetCDFforPoints(self, geopointList = [], startDate = None,
                                             endDate = None, months = []):
        '''
        months - months of year over which the average is taken
        '''
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



    def getSpatialMeanDataFromNetCDFforPoints(self, geopointList = [], startDate = None,
                                             endDate = None, months = []):
        """
        months - months of year over which the average is taken
        """
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
        data = [[]]
        record_length = 4
        f = open(path)
        for line in f:
            line = line.rstrip()
            if len(data[-1]) == self.nEastWest:
                data.append([])
            for j in xrange(0,len(line), record_length):
                data[-1].append(float(line[j:j + record_length].strip()))
        data = np.array(data)
        print data.shape
        f.close()
        return data.transpose()

    def convertAllDataToNetcdf(self, pathToNetCDF = 'data/swe_ross_brown/swe.nc'):

        ds = Dataset(pathToNetCDF, 'w', format = 'NETCDF3_CLASSIC')
        

        ds.createDimension('time', None)


        nLon = None
        nLat = None

        nDates = len(self.all_dates)
        for i, theDate in enumerate(self.all_dates):
            filePath = os.path.join(self.root_path, theDate.strftime(self.file_name_format))
            data = self._readFromTxtFile(filePath)
            if nLon is None:
                nLon, nLat = data.shape
                ds.createDimension('lon', nLon)
                ds.createDimension('lat', nLat)
                lonVariable = ds.createVariable('longitude', 'f4', ('lon', 'lat'))
                latVariable = ds.createVariable('latitude', 'f4', ('lon', 'lat'))

                lonLats = self.get2DLonsAndLats()
                lonVariable[:] = lonLats[0][:, :]
                latVariable[:] = lonLats[1][:, :]

                
                sweVariable = ds.createVariable('SWE', 'f4', ('time', 'lon', 'lat'))
                sweVariable.units = 'mm of equivalent water'

            sweVariable[i, :, :] = data[:, :] * self.conversionCoef
            print ' {0} / {1} '.format(i, nDates)

        timeVariable = ds.createVariable('time', 'i4', ('time'))

        ncSinceFormat = '%Y-%m-%d %H:%M:%S'
        timeVariable.units = ' hours since ' + self.getStartDate().strftime(ncSinceFormat)
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
            result[d] = np.mean(data[i_indices, j_indices]) * self.conversionCoef
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

    def plot(self, path = ''):
        data = self._readFromTxtFile(path)
        plt.figure()
        plt.pcolormesh(data)
        plt.title(os.path.basename(path))
        plt.colorbar()
    pass

def test():
    application_properties.set_current_directory()
    swe = SweHolder()

    #swe.convertAllDataToNetcdf()

#    swe.plot(path = 'data/swe_ross_brown/derksen/1985042312')
#    swe.plot(path = 'data/swe_ross_brown/derksen/1985010112')
#    swe.plot(path = 'data/swe_ross_brown/derksen/1985080112')
#    plt.show()

if __name__ == "__main__":
    test()
    print "Hello World"
