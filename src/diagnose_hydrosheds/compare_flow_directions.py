
__author__="huziy"
__date__ ="$Aug 8, 2011 10:29:34 AM$"

from data.cell import Cell
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import application_properties

import matplotlib.pyplot as plt
import netCDF4 as nc

import numpy as np
from data import direction_and_value


def init_cells(direction_map):
    cells = []
    nx, ny = direction_map.shape
    for i in xrange(nx):
        cells.append([])
        for j in xrange(ny):
            theCell = Cell()
            theCell.x = i
            theCell.y = j
            cells[-1].append(Cell())

    for i in xrange(nx):
        for j in xrange(ny):
            i_next, j_next = direction_and_value.to_indices(i, j, direction_map[i,j])
            if i_next < 0 or j_next < 0 or i_next == nx or j_next == ny:
                nextCell = None
            else:
                nextCell = cells[i_next][j_next]
            cells[i][j].set_next(nextCell)

    return cells



def calculate_accumulation_index(cells):
    nx, ny = len(cells), len(cells[0])
    result = np.zeros((nx, ny))
    for i in xrange(nx):
        for j in xrange(ny):
            result[i, j] = cells[i][j].calculate_number_of_upstream_cells()
    return result


def compare():
    path_hydrosheds = 'data/hydrosheds/africa_044/w001001.adf'
    #path_huziy = 'data/hydrosheds/directions_af_parallel.nc'
    path_huziy = "data/hydrosheds/directions_africa_dx0.44deg.nc"
    data_huziy = nc.Dataset(path_huziy).variables['flow_direction_value'][:]
    data_huziy = data_huziy[21:215, 20:221]

    print data_huziy.shape
    print data_huziy.min(), data_huziy.max()
    

    dataset = gdal.Open(path_hydrosheds, gdalconst.GA_ReadOnly)
    data = dataset.ReadAsArray()

    print 'Driver: ', dataset.GetDriver().ShortName,'/', \
          dataset.GetDriver().LongName
    print 'Size is ',dataset.RasterXSize,'x',dataset.RasterYSize, \
          'x',dataset.RasterCount
    print 'Projection is ',dataset.GetProjection()

    geotransform = dataset.GetGeoTransform()
    if not geotransform is None:
        print 'Origin = (',geotransform[0], ',',geotransform[3],')'
        print 'Pixel Size = (',geotransform[1], ',',geotransform[5],')'


    print data.shape
    data = np.flipud(data).transpose()
    print 'Calculating HS acc index'
    acc_index_HS = calculate_accumulation_index(init_cells(data))
    print 'Calculating HU acc index'
    acc_index_HU = calculate_accumulation_index(init_cells(data_huziy))




    plt.figure()
    plt.pcolormesh(np.ma.masked_where(data <= 0, data))
    plt.colorbar()
    plt.title('Hydrosheds')


    plt.figure()
    plt.pcolormesh(np.ma.masked_where(data_huziy <= 0, data_huziy))
    plt.colorbar()
    plt.title('Huziy data')





    condition = (data > 0) & (data_huziy > 0)
    print data[condition].shape
    plt.figure()
    plt.scatter(acc_index_HU[condition], acc_index_HS[condition])
    plt.xlabel('Huziy')
    plt.ylabel('Hydrosheds')
    plt.plot([0,1400], [0, 1400], color = 'k')
    plt.title('accumulation index')

    print 'Number of equal points ', sum(map(int, data[condition] == data_huziy[condition]))
    print 'Number of different points ', sum(map(int, data[condition] != data_huziy[condition]))



   
    plt.figure()
    plt.pcolormesh(np.ma.masked_where(acc_index_HS == 0, acc_index_HS).transpose())
    plt.colorbar()
    plt.title('Hydrosheds acc')


    plt.figure()
    plt.pcolormesh(np.ma.masked_where(acc_index_HU == 0, acc_index_HU).transpose())
    plt.colorbar()
    plt.title('acc index Huziy')


    plt.figure()
    condition = (acc_index_HU > 0) & (acc_index_HS > 0)
    delta = np.zeros_like(acc_index_HU)
    delta[condition] = (acc_index_HU[condition] - acc_index_HS[condition]) / acc_index_HU[condition]
    plt.pcolormesh(np.ma.masked_where(delta == 0, delta).transpose())
    plt.colorbar()
    plt.title('(Huziy-HydroSheds) / Huziy')



    plt.show()





    print np.min(data), np.max(data)
    print data.shape
    print type(data)


if __name__ == "__main__":
    application_properties.set_current_directory()
    compare()
    print "Hello World"
