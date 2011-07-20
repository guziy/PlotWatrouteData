
__author__="huziy"
__date__ ="$27 mai 2010 09:49:54$"

from data.datapoint import DataPoint

class ModelPoint(DataPoint):
    '''
    Represents one station data timeseries
    '''
    def __init__(self, ix = None, iy = None, id = None):
        DataPoint.__init__(self)
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



    def __str__(self):
        result = 'grid point: '
        if self.ix != None:
            result += 'ix = ' + str( self.ix )
            result += ', '
        if self.iy != None:
            result += 'iy = ' + str( self.iy )
        if self._longitude != None:
            result += '(lon, lat) = (%f, %f); ' % (self._longitude, self._latitude)
        if self.id != None:
            result += self.id
        return result



if __name__ == "__main__":
    print "Hello World"
