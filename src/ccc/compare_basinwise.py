from datetime import datetime
import os
import pickle
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import application_properties
from util import plot_utils

__author__ = 'huziy'

import numpy as np
import descartes
from data.cru_data_reader import CruReader
from shape import read_shape_file
import ccc_analysis

from netCDF4 import Dataset
import matplotlib.pyplot as plt

def compare_basin_wise(obs_data, obs_mask, crcm4_data, crcm4_mask, basin_polygons_map = None):
    """
    :type obs_data: numpy.array(month, lon, lat)
    :type obs_mask: dict
    :type crcm4_data: numpy.array(month, lon, lat)
    :type crcm4_mask: dict
    """
    basin_names = obs_mask.keys()
    if basin_polygons_map is not None:
        #sort from north to south
        basin_names.sort(key=lambda x: basin_polygons_map[x].centroid.y, reverse=True)

    n_basins = len(basin_names)

    plot_utils.apply_plot_params(width_pt=None, width_cm=30, font_size=12)
    fig = plt.figure()
    assert isinstance(fig, Figure)

    ncols = 5
    nrows = n_basins // ncols + int( n_basins % ncols != 0 )

    gs = gridspec.GridSpec(nrows, ncols)
    for i, name in enumerate( basin_names ):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col])
        assert isinstance(ax, Axes)
        obs = np.mean(obs_data[:,obs_mask[name] == 1], axis=1)
        mod = np.mean(crcm4_data[:,crcm4_mask[name] == 1], axis=1) - 273.15
        diff = np.array(mod - obs)
        diff = np.insert(diff, 0, diff[-1])

        ax.plot(xrange(0,obs_data.shape[0] + 1), diff, linewidth = 3)
#        h_obs = ax.plot(xrange(1,obs_data.shape[0] + 1),
#            np.mean(obs_data[:,obs_mask[name] == 1], axis=1),
#            "r", linewidth = 2
#        )
#        h_mod = ax.plot(xrange(1, crcm4_data.shape[0] + 1),
#            np.mean(crcm4_data[:,crcm4_mask[name] == 1], axis=1) - 273.15,
#            "b", linewidth = 2
#        )

        x_limits = ax.get_xlim()
        ax.plot(x_limits, [0] * 2, "k")

        ax.set_title(name)
        ax.yaxis.set_ticks(xrange(-5, 6, 2))
        ax.xaxis.set_ticks(xrange(1,13,3))
        ax.xaxis.set_ticklabels(["Jan", "Apr", "Jul", "Nov"])


    gs.update(wspace = 0.3, hspace = 0.5)
    fig.suptitle("T(2m), CRCM4 - CRU, basin averages")
    fig.savefig("basinwise_comparison.png")









    pass

def get_basin_masks_from_file(path = "data/infocell/amno180x172_basins.nc"):
    """
    gets masks for basins from netcdf file
    """
    ds = Dataset(path)
    result = {}
    for name, mask_var in ds.variables.iteritems():
        result[name] = mask_var[:]
    ds.close()
    return result


def get_point(x, y, point_obj):
    """
    :type point_obj: Point
    """
    point_obj.coords = [(x,y)]
    return point_obj


def main():

    start_date = datetime(1970, 1, 1)
    end_date = datetime(1999, 12, 31)

    #compare 2m temperatures
    cache_file = "compare_basinwise.bin"
    if not os.path.isfile(cache_file):
        cru = CruReader(path="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.tmp.dat.nc",
            var_name="tmp", create_tree_for_interpolation=False)
        cru_t2m = cru.get_monthly_normals(start_date = start_date, end_date = end_date)
        data_store = {"cru_t2m": cru_t2m, "cru_lons": cru.lons_2d, "cru_lats": cru.lats_2d}

        crcm4_t2m = ccc_analysis.get_monthly_normals("data/ccc_data/aex/aex_p1st",
            start_date = start_date, end_date = end_date
        )
        data_store["crcm4_t2m"] = crcm4_t2m


        #create basin masks
        #read basins polygons
        basins_map = read_shape_file.get_basins_as_shapely_polygons()
        data_store["basin_polygons_map"] = basins_map

        print( "created basin polygons from the shape file." )
        #obs mask
        obs_mask = {}
        cru_lons1d = cru.lons_2d.flatten()
        cru_lats1d = cru.lats_2d.flatten()
        p = Point()
        for basin_name, basin_poly in basins_map.iteritems():
            assert isinstance(basin_poly, Polygon)
            the_mask = map( lambda x: int( basin_poly.intersects(get_point(x[0], x[1], p))), zip(cru_lons1d, cru_lats1d))
            the_mask = np.reshape( np.array(the_mask), cru.lons_2d.shape )
            obs_mask[basin_name] = the_mask
            print( "created mask for {0}".format( basin_name ))
        data_store["obs_mask"] = obs_mask

        #model mask
        data_store["crcm4_mask"] = get_basin_masks_from_file()

        pickle.dump(data_store, open(cache_file, "wb"))
    else:
        data_store = pickle.load(open(cache_file))

    #plot data
    compare_basin_wise(data_store["cru_t2m"], data_store["obs_mask"],
        data_store["crcm4_t2m"], data_store["crcm4_mask"], data_store["basin_polygons_map"]
    )

    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  