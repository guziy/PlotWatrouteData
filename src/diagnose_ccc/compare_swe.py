#Compare CRCM4 swe data with Ross Brown's dataset
#
__author__="huziy"
__date__ ="$Aug 5, 2011 10:44:05 AM$"

from data.swe_ross_brown import SweHolder
from util.geo.GeoPoint import GeoPoint
from netCDF4 import Dataset
from ccc.ccc import champ_ccc
import application_properties
from datetime import timedelta
from datetime import datetime
import numpy as np

import os

from plot2D.map_parameters import polar_stereographic

import matplotlib.pyplot as plt
import matplotlib as mpl

##1. Read mask for a basin
##2. Read ccc model data for the basin
##3. Read measure data correspondng to the selected ccc points
##4. Compare evolution of averages in time

import util.plot_utils as plot_utils


def getBasinMask(basinName = 'RDO'):
    '''
    returns a mask for a given basin
    '''
    path = 'data/infocell/amno180x172_basins.nc'
    ds = Dataset(path)
    mask = ds.variables[basinName][:]
    ds.close()
    return mask


def getSelectedBasinNames():
    return ['BAL', 'RDO', 'LGR']


def getCCCDataForMask(mask = None, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
                    startDate = None, endDate = None):

    '''
    returns a map {date => mean value over a mask}
    '''

    dt = timedelta(hours = 6)
    dates = []
    result = {}
    for fName in os.listdir(path_to_ccc):
        fPath = os.path.join( path_to_ccc, fName )
        cccObj = champ_ccc(fPath)
        theDate = datetime.strptime( fName.split('_')[-1][:-4], '%Y%m')

        if endDate != None and startDate != None:
            if endDate < theDate or theDate < startDate:
                continue

        dataStore = cccObj.charge_champs()
        for field in dataStore:
            if startDate != None and endDate != None:
                if startDate > theDate:
                    continue
                if endDate < theDate:
                    break

            
            fieldData = field['field']
            dates.append(theDate)
            result[theDate] = np.mean(fieldData[mask == 1])
            theDate += dt
    dates.sort()
    return dates, [result[d] for d in dates]
    pass


def test():
    start = datetime(1985,01,01,00)
    end = datetime(1986, 12, 31,00)
    basinName = 'RDO'
    mask = getBasinMask(basinName)
    print getCCCDataForMask(mask, startDate = start, endDate = end)



def toStampYear(theDate, stamp_year = 2000):
    '''
    stamp_year should be aleap year, if not problems may arise for 29/02
    '''
    try:
        return datetime(stamp_year, theDate.month, theDate.day, theDate.hour, theDate.minute)
    except ValueError, e:
        print e
        print stamp_year
        print theDate


def compare_means(basinName = 'RDO'):
    lons = polar_stereographic.lons
    lats = polar_stereographic.lats

    mask = getBasinMask(basinName)

    start = datetime(1980,01,01,00)
    end = datetime(1996, 12, 31,00)

    lons_selected = lons[ mask == 1 ]
    lats_selected = lats[ mask == 1 ]
    points = [GeoPoint(longitude = lon, latitude = lat) for lon, lat in zip(lons_selected, lats_selected)]

    sweObs = SweHolder()
    obsData = sweObs.getMeanDataFromNetCDFforPoints(points, startDate = start, endDate = end)
    print 'finished reading observations'
    modelData = getCCCDataForMask(mask, startDate = start, endDate = end)
    print 'finished reading model data'
    print 'finished reading input mean timeseries'

    stamp_year = 2000
    obsStamp = map(lambda x: toStampYear(x, stamp_year = stamp_year), obsData[0])
    modelStamp = map(lambda x: toStampYear(x, stamp_year = stamp_year), modelData[0])
    print 'calculated stamp dates'

    ##calculate mean for a day of year
    obsDict = {}
    for stampDate, value in zip(obsStamp, obsData[1]):
        if not obsDict.has_key(stampDate):
            obsDict[stampDate] = []
        obsDict[stampDate].append(value)

    for key, theList in obsDict.iteritems():
        obsDict[key] = np.mean(theList)

    obsDates = sorted(obsDict)
    obsMeanValues = [obsDict[d] for d in obsDates]



    #do the same thing as for obs for the model data
    modelDict = {}
    for stampDate, value in zip(modelStamp, modelData[1]):
        if not modelDict.has_key(stampDate):
            modelDict[stampDate] = []
        modelDict[stampDate].append(value)


    for key, theList in modelDict.iteritems():
        modelDict[key] = np.mean(theList)

    modelDates = sorted(modelDict)
    modelMeanValues = [modelDict[d] for d in modelDates]

    print 'Calculated mean for day of year and over a basin points'


    plt.figure()
    plt.title(basinName + ': SWE')
    plt.plot(obsDates, obsMeanValues, label = 'Obs.', color = 'red', lw = 3)
    plt.plot(modelDates, modelMeanValues, label = 'Model', color = 'blue', lw = 3)
    plt.ylabel('mm')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(
        mpl.dates.MonthLocator(bymonth = range(2,13,2))
    )
    plt.legend()
    plt.savefig(basinName + '_swe.pdf', bbox_inches = 'tight')



def compare(basinName = 'RDO'):
    lons = polar_stereographic.lons
    lats = polar_stereographic.lats

    mask = getBasinMask(basinName)

    start = datetime(1985,01,01,00)
    end = datetime(1986, 12, 31,00)

    lons_selected = lons[ mask == 1 ]
    lats_selected = lats[ mask == 1 ]
    points = [GeoPoint(longitude = lon, latitude = lat) for lon, lat in zip(lons_selected, lats_selected)]

    sweObs = SweHolder()
    obsData = sweObs.getMeanDataFromNetCDFforPoints(points, startDate = start, endDate = end)
    modelData = getCCCDataForMask(mask, startDate = start, endDate = end)

    plt.figure()
    plt.title('SWE')
    plt.plot(obsData[0], obsData[1], label = 'Obs.', color = 'red', lw = 3)
    plt.plot(modelData[0], modelData[1], label = 'Model', color = 'blue', lw = 3)
    plt.ylabel('mm')
    plt.legend()
    plt.show()



def getBasinNames():
    path = 'data/infocell/amno180x172_basins.nc'
    ds = Dataset(path)
    names = ds.variables.keys()
    ds.close()
    return names


if __name__ == "__main__":
    application_properties.set_current_directory()
    #test()
    #compare_means('BEL')
    
    print getBasinNames()
    plot_utils.apply_plot_params(width_pt = 400, font_size = 15)

    basinNames = getSelectedBasinNames()
    for basinName in basinNames:
        compare_means(basinName = basinName)
        print basinName
     
    
    print "Hello World"
