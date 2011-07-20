
__author__="huziy"
__date__ ="$18 mai 2010 11:56:35$"

from data.datapoint import DataPoint
from datetime import *

class Station(DataPoint):
    '''
    Represents one station, contains data timeseries
    '''
    def __init__(self, id = None, longitude = None, latitude = None):
        DataPoint.__init__(self)
        self.id = id
        self.longitude = longitude
        self.latitude = latitude


    def __str__(self):
        result = 'measure station: '
        if self.id != None:
            result += 'id = ' + self.id
            result += ', '
        if self.longitude != None:
            result += 'longitude = %3.2f' % self.longitude
            result += ', '
        if self.latitude != None:
            result += 'latitude = %3.2f' % self.latitude 
        return result







if __name__ == "__main__":
    print "Hello World"
