
__author__="huziy"
__date__ ="$27 mai 2010 09:49:54$"

from data.datapoint import DataPoint

class ModelPoint(DataPoint):
    '''
    Represents one point data timeseries
    '''
    def __init__(self, dates = None, values = None, ix = None, iy = None, id = None):
        DataPoint.__init__(self,dates, values)
        self.ix = ix
        self.iy = iy
        self._longitude = None
        self._latitude = None
        self.id = id

    def set_longitude(self, longitude):
        self._longitude = longitude

    def get_longitude(self):
        return self._longitude

    def set_latitude(self, latitude):
        self._latitude = latitude

    def get_latitude(self):
        return self._latitude


    def clear_timeseries(self):
        self.timeseries = {}
        self.sorted_dates = []



    def __str__(self):
        result = 'grid point: '
        if self.ix is not None:
            result += 'ix = ' + str( self.ix )
            result += ', '
        if self.iy is not None:
            result += 'iy = ' + str( self.iy )
        if self._longitude is not None:
            result += '(lon, lat) = (%f, %f); ' % (self._longitude, self._latitude)
        if self.id is not None:
            result += self.id
        return result



if __name__ == "__main__":
    print "Hello World"
