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

from readers import read_infocell

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

from data import cehq_station
from plot2D import calculate_performance_errors as cpe

def getBasinMask(basinName = 'RDO'):
    """
    returns a mask for a given basin
    """
    path = 'data/infocell/amno180x172_basins.nc'
    ds = Dataset(path)
    mask = ds.variables[basinName][:]
    ds.close()
    return mask


def get_domain_mask(path = 'data/infocell/amno180x172_basins.nc'):
    ds = Dataset(path)
    result = None
    for v in ds.variables.values():
        if result is None:
            result = v[:]
        else:
            result += v[:]
    ds.close()
    return result


def getSelectedBasinNames():
    return ['BAL', 'RDO', 'LGR']



def getTemporalMeanCCCDataForMask(mask=None, path_to_ccc='data/ccc_data/aex/aex_p1sno',
                                  startDate=None, endDate=None,
                                  months=None):

    """
    months - is a list of month indices (i.e. 1,2,3,4 .. 12), over which
    the mean is calculated
    returns
    """
    if not months: months = []

    dt = timedelta(hours = 6)
    dates = []
    result = {}
    for fName in os.listdir(path_to_ccc):
        fPath = os.path.join( path_to_ccc, fName )
        cccObj = champ_ccc(fPath)
        theDate = datetime.strptime( fName.split('_')[-1][:-4], '%Y%m')

        if endDate is not None and startDate is not None:
            if endDate < theDate or theDate < startDate:
                continue

        dataStore = cccObj.charge_champs()
        for field in dataStore:
            if startDate is not None and endDate is not None:
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



class DateAndPath:
    """
    Holds a start date and path corresponding to a ccc file
    """
    def __init__(self, date = None, path = ""):
        self.date = date
        self.path = path


def getMonthlyNormalsAveragedOverMask(mask = None, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
                    startDate = None, endDate = None):
    """
    returns dates of the stamp year and normals corresponding to the dates
    """

    times, data = getSpatialIntegralCCCDataForMask(mask = mask, path_to_ccc = path_to_ccc,
                                                   startDate= startDate, endDate=endDate)

    monthly_normals = np.zeros((12,))
    for t, x in zip(times, data):
        monthly_normals[t.month - 1] += x

    dt = times[-1] - times[0]
    n_years = float(dt.days // 365)
    return monthly_normals / n_years
    pass


def getDailyNormalsAveragedOverMask(mask = None, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
                    startDate = None, endDate = None):
    """
    returns dates of the stamp year and normals corresponding to the dates
    """

    times, data = getSpatialIntegralCCCDataForMask(mask = mask, path_to_ccc = path_to_ccc)

    year = 2001
    normals = {}
    for t, x in zip(times, data):
        #skip the 29th of february
        if t.month == 2 and t.day == 29: continue
        d = datetime(year, t.month, t.day, 0, 0, 0)
        if normals.has_key(d):
            normals[d] += [x]
        else:
            normals[d] = [x]
    
    ds = normals.keys()
    ds.sort()
    return ds, [np.mean(normals[d]) for d in ds]

    pass

def getSpatialIntegralCCCDataForMask(mask = None, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
                    startDate = None, endDate = None):

    """
    returns 2 lists of the same dimensions one contains dates and the other,
    integral values over the mask corresponding to the date
    """

    dt = timedelta(hours = 6)
    dates = []
    result = []



    monthlyFileNames = os.listdir(path_to_ccc)
    filePaths = map(lambda x: os.path.join(path_to_ccc, x), monthlyFileNames)
    the_func = lambda x: datetime.strptime( x.split('_')[-1][:-4], '%Y%m')
    monthlyDates = map(the_func, monthlyFileNames) # get dates for months

    the_zip = zip(monthlyDates, filePaths)
    dps = [DateAndPath(date = the_date, path = the_path) for the_date, the_path in the_zip]
    dps.sort(key =  lambda x: x.date)

    for dp in dps:
        fPath = dp.path
        cccObj = champ_ccc(fPath)
        theDate = dp.date

        if endDate is not None and startDate is not None:
            if endDate < theDate or theDate < startDate:
                continue

        dataStore = cccObj.charge_champs()
        for field in dataStore:
            if startDate is not None and endDate is not None:
                if startDate > theDate:
                    continue
                if endDate < theDate:
                    break


            fieldData = field['field']
            dates.append(theDate)
            result.append(np.sum(fieldData[mask == 1]))
            theDate += dt

    return dates, np.array(result) * dt.seconds
    pass


def getSpatialMeanCCCDataForMask(mask = None, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
                    startDate = None, endDate = None):

    """
    returns a map {date => mean value over a mask}
    """

    dt = timedelta(hours = 6)
    dates = []
    result = {}
    for fName in os.listdir(path_to_ccc):
        fPath = os.path.join( path_to_ccc, fName )
        cccObj = champ_ccc(fPath)
        theDate = datetime.strptime( fName.split('_')[-1][:-4], '%Y%m')

        if endDate is not None and startDate is not None:
            if endDate < theDate or theDate < startDate:
                continue

        dataStore = cccObj.charge_champs()
        for field in dataStore:
            if startDate is not None and endDate is not None:
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
    """
    stamp_year should be aleap year, if not problems may arise for 29/02
    """
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
    """
    timeseries not changed, averaging over the points where mask == 1
    """
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




swe_fig = None
def compare_daily_normals_integral_over_mask(mask = None, start = None, end = None, label = '',
                                             subplot_count = None, subplot_total = 10
                                             ):
    """

    """
    lons = polar_stereographic.lons
    lats = polar_stereographic.lats

    lons_selected = lons[ mask == 1 ]
    lats_selected = lats[ mask == 1 ]

    global swe_fig

    if subplot_count == 1:
        swe_fig = plt.figure()
        plot_utils.apply_plot_params(font_size=25, width_pt=900, aspect_ratio=2.5)
        swe_fig.subplots_adjust(hspace = 0.6, wspace = 0.2, top = 0.9)



    points = [GeoPoint(longitude = lon, latitude = lat) for lon, lat in zip(lons_selected, lats_selected)]

    sweObs = SweHolder()
    obsData = sweObs.getSpatialIntegralFromNetcdfForPoints(points, startDate = start, endDate = end)
    print 'finished reading observations'
    modelData = getSpatialIntegralCCCDataForMask(mask = mask, path_to_ccc = 'data/ccc_data/aex/aex_p1sno',
                                                 startDate = start, endDate = end)
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

    print 'Calculated mean for day of year and over selected points'


    plt.title('Upstream of {0}'.format(label))
    line1 = plt.plot(modelDates, modelMeanValues, color = 'blue', lw = 3)
    line2 = plt.plot(obsDates, obsMeanValues, color = 'red', lw = 3)

    #plt.ylabel('mm')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(
        mpl.dates.MonthLocator(bymonth = range(2,13,2))
    )


    if subplot_count == subplot_total:
         lines = (line1, line2)
         labels = ("CRCM4", "CRU")
         swe_fig.legend(lines, labels, 'upper center')
         swe_fig.text(0.05, 0.5, 'SWE (mm)',
                       rotation=90,
                       ha = 'center', va = 'center'
                        )
         swe_fig.savefig("swe.pdf")




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
    """
    Compare seasonal mean
    """
    start = datetime(1980,01,01,00)
    end = datetime(1996, 12, 31,00)
    months = [12,1,2]
    #calculate mean for ccc data accounting for the mask
    domain_mask = get_domain_mask()
    cccData = getTemporalMeanCCCDataForMask(domain_mask,
                                            startDate = start,
                                            endDate = end, months = months)
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
    basemap.pcolormesh(xs, ys, to_plot, cmap = mpl.cm.get_cmap('jet', 9), vmin = -60, vmax = 120)
    basemap.drawcoastlines()
    plt.colorbar(ticks = LinearLocator(numticks = 10), format = '%.1f')
    plt.title('Snow Water Equivalent (%) \n $(S_{\\rm CRCM4} - S_{\\rm obs.})/S_{\\rm obs.}\\cdot100\%$\n')

    #zoom to domain
    selected_x = xs[~to_plot.mask]
    selected_y = ys[~to_plot.mask]
    marginx = abs(np.min(selected_x) * 5.0e-2)
    marginy = abs(np.min(selected_y) * 5.0e-2)

    plt.xlim(np.min(selected_x) - marginx, np.max(selected_x) + marginx)
    plt.ylim(np.min(selected_y) - marginy, np.max(selected_y) + marginy)

    bb.plot_basin_boundaries_from_shape(basemap, plotter = plt, linewidth = 2, edge_color = 'k')



    ##overlay flow directions
    cells  = cpe.get_connected_cells('data/hydrosheds/directions_for_streamflow9.nc')

    read_infocell.plot_directions(cells, basemap = basemap, domain_mask = get_domain_mask())


    selected_stations = [
            104001, 103715, 93806, 93801,
            92715, 81006, 61502, 80718,
            40830, 42607
    ]
    selected_ids = map(lambda x: "%06d" % x, selected_stations)
    print selected_ids

    cpe.plot_station_positions(id_list=selected_ids, use_warpimage=False,
                               save_to_file=False, the_basemap=basemap)

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
