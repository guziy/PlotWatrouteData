#Compare CRCM4 swe data with Ross Brown's dataset
#
__author__="huziy"
__date__ ="$Aug 5, 2011 10:44:05 AM$"

from matplotlib.ticker import LinearLocator
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

import matplotlib as mpl
import shape.basin_boundaries as bb

def getBasinMask(basinName = 'RDO'):
    '''
    returns a mask for a given basin
    '''
    path = 'data/infocell/amno180x172_basins.nc'
    ds = Dataset(path)
    mask = ds.variables[basinName][:]
    ds.close()
    return mask


def get_domain_mask(path = 'data/infocell/amno180x172_basins.nc'):
    ds = Dataset(path)
    result = None
    for v in ds.variables.values():
        if result == None:
            result = v[:]
        else:
            result += v[:]
    ds.close()
    return result


def getSelectedBasinNames():
    return ['BAL', 'RDO', 'LGR']



def getTemporalMeanCCCDataForMask(mask = None, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
                    startDate = None, endDate = None, months = []):

    '''
    months - is a list of month indices (i.e. 1,2,3,4 .. 12), over which
    the mean is calculated
    returns
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

            if theDate.month not in months:
                continue

            fieldData = field['field']
            dates.append(theDate)
            result[theDate] = fieldData[mask == 1]
            theDate += dt

    x = np.array(result.values())
    return np.mean(x, axis = 0)



def getSpatialMeanCCCDataForMask(mask = None, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
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
    print getSpatialMeanCCCDataForMask(mask, startDate = start, endDate = end)



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


def compare_means(basinName = 'RDO',
                  start = datetime(1980,01,01,00),
                  end = datetime(1996, 12, 31,00)):

    mask = getBasinMask(basinName)
    compare_daily_normals_mean_over_mask(mask, start = start, end = end, label = basinName)




def compare_spatial_means_over_mask(mask = None, label = ''):
    '''
    timeseries not changed, averaging over the points where mask == 1
    '''
    start = datetime(1980,01,01,00)
    end = datetime(1996, 12, 31,00)

    lons = polar_stereographic.lons
    lats = polar_stereographic.lats

    lons_selected = lons[ mask == 1 ]
    lats_selected = lats[ mask == 1 ]
    points = [GeoPoint(longitude = lon, latitude = lat) for lon, lat in zip(lons_selected, lats_selected)]

    sweObs = SweHolder()
    obsData = sweObs.getSpatialMeanDataFromNetCDFforPoints(points, startDate = start, endDate = end)
    modelData = getSpatialMeanCCCDataForMask(mask, startDate = start, endDate = end)

    plt.figure()
    plt.title('SWE' + ' ' + label)
    plt.plot(obsData[0], obsData[1], label = 'Obs.', color = 'red', lw = 3)
    plt.plot(modelData[0], modelData[1], label = 'Model', color = 'blue', lw = 3)
    plt.ylabel('mm')
    plt.legend()
    plt.savefig('swe_compare_{0}.pdf'.format(label), bbox_inches = 'tight')


def compare_daily_normals_mean_over_mask(mask = None, start = None, end = None, label = ''):
    lons = polar_stereographic.lons
    lats = polar_stereographic.lats

    lons_selected = lons[ mask == 1 ]
    lats_selected = lats[ mask == 1 ]

    points = [GeoPoint(longitude = lon, latitude = lat) for lon, lat in zip(lons_selected, lats_selected)]

    sweObs = SweHolder()
    obsData = sweObs.getSpatialMeanDataFromNetCDFforPoints(points, startDate = start, endDate = end)
    print 'finished reading observations'
    modelData = getSpatialMeanCCCDataForMask(mask, startDate = start, endDate = end)
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


    plt.figure(figsize = (8, 6), dpi = 80)
    plt.title('SWE {0}'.format(label))
    plt.plot(obsDates, obsMeanValues, label = 'Obs.', color = 'red', lw = 3)
    plt.plot(modelDates, modelMeanValues, label = 'Model', color = 'blue', lw = 3)
    plt.ylabel('mm')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(
        mpl.dates.MonthLocator(bymonth = range(2,13,2))
    )
    plt.legend()
    plt.savefig(label + '_swe.pdf', bbox_inches = 'tight')


def compare(basinName = 'RDO'):
    mask = getBasinMask(basinName)
    compare_spatial_means_over_mask(mask = mask)



def getBasinNames():
    path = 'data/infocell/amno180x172_basins.nc'
    ds = Dataset(path)
    names = ds.variables.keys()
    ds.close()
    return names



def compare_swe_2d():
    '''
    Compare seasonal mean
    '''
    start = datetime(1980,01,01,00)
    end = datetime(1996, 12, 31,00)
    months = [2,3,4]
    #calculate mean for ccc data accounting for the mask
    domain_mask = get_domain_mask()
    cccData = getTemporalMeanCCCDataForMask(domain_mask,
                                            startDate = start, endDate = end, months = months)
    lon_selected = polar_stereographic.lons[domain_mask == 1]
    lat_selected = polar_stereographic.lats[domain_mask == 1]

    geopointList = []
    for lon, lat in zip(lon_selected, lat_selected):
        geopointList.append(GeoPoint(longitude = lon, latitude = lat))

    print 'Selecting obs data'
    sweObs = SweHolder()
    obsData = sweObs.getTemporalMeanDataFromNetCDFforPoints(geopointList, startDate = start,
                                                            endDate = end, months = months)

    

    to_plot = np.ma.masked_all(polar_stereographic.lons.shape)
    condition = domain_mask == 1
    to_plot[condition] = (cccData - obsData) / obsData * 100
    xs = polar_stereographic.xs
    ys = polar_stereographic.ys
    basemap = polar_stereographic.basemap

    plot_utils.apply_plot_params()
    basemap.pcolormesh(xs, ys, to_plot, cmap = mpl.cm.get_cmap('jet', 7))
    basemap.drawcoastlines()
    plt.colorbar(ticks = LinearLocator(numticks = 8), format = '%.1f')
    plt.title('Snow Water Equivalent (%) \n $(S_{\\rm CRCM4} - S_{\\rm obs.})/S_{\\rm obs.}\\cdot100\%$\n')

    #zoom to domain
    selected_x = xs[~to_plot.mask]
    selected_y = ys[~to_plot.mask]
    marginx = abs(np.min(selected_x) * 5.0e-2)
    marginy = abs(np.min(selected_y) * 5.0e-2)

    plt.xlim(np.min(selected_x) - marginx, np.max(selected_x) + marginx)
    plt.ylim(np.min(selected_y) - marginy, np.max(selected_y) + marginy)

    bb.plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 2, edge_color = 'k')
    plt.savefig('swe_djf_validation.pdf', bbox_inches = 'tight')


    pass


def compare_averages_over_basins():
    print getBasinNames()
    plot_utils.apply_plot_params(width_pt = 400, font_size = 15)

    basinNames = getSelectedBasinNames()
    for basinName in basinNames:
        compare_means(basinName = basinName)
        print basinName


if __name__ == "__main__":
    application_properties.set_current_directory()
    #test()
    #compare_means('BEL')
    compare_swe_2d()
    
     
    
    print "Hello World"
