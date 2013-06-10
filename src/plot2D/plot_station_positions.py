import itertools
from matplotlib.font_manager import FontProperties
from scipy.spatial.kdtree import KDTree
from data import cehq_station, data_select
from data.cehq_station import Station
from plot2D.map_parameters import polar_stereographic
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'
import matplotlib.pyplot as plt
from shape import basin_boundaries

def main():

    start_year = 1970
    end_year = 1999


    stations = cehq_station.read_station_data(folder="data/cehq_measure_data_all")
    stations = list( itertools.ifilter( lambda s: s.is_natural, stations) )
    for s in stations:
        s.delete_data_after_year(end_year)
        s.delete_data_before_year(start_year)
        pass


    stations = list( itertools.ifilter(lambda s: s.get_num_of_years_with_continuous_data() >= 10, stations) )
    s = stations[0]

    assert isinstance(s, Station)





    #stations = list( itertools.ifilter(lambda s: s.is_natural, stations) )

    x, y = polar_stereographic.lons, polar_stereographic.lats
    basemap = polar_stereographic.basemap
    x, y = basemap(x,y)

    sx = [s.longitude for s in stations]
    sy = [s.latitude for s in stations]

    sx, sy = basemap(sx, sy)

    #read model data
    model_file_path = "data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc"
    acc_area = data_select.get_field_from_file(path=model_file_path,
        field_name="drainage")
    i_indices, j_indices = data_select.get_indices_from_file(path=model_file_path)
    lons_1d = data_select.get_field_from_file(path=model_file_path, field_name="longitude")
    lats_1d = data_select.get_field_from_file(path=model_file_path, field_name="latitude")

    x1d, y1d, z1d = lat_lon.lon_lat_to_cartesian(lons_1d, lats_1d)
    kdtree = KDTree(zip(x1d, y1d, z1d))

    print "Id: 4 DA (km2) <-> 4 dist (km) <-> 4 (i,j)"
    #basemap.scatter(sx, sy, c = "r", zorder = 5)
    for s, isx, isy in zip( stations, sx, sy ):
        assert isinstance(s, Station)
        plt.annotate(s.id, xy=(isx, isy),
                 bbox = dict(facecolor = 'white'), weight = "bold", font_properties = FontProperties(size=0.5))

        #get model drainaige areas for the four closest gridcells to the station
        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
        dists, indices = kdtree.query([x0, y0, z0], k = 4)
        dists /= 1000
        print("{0}: {1:.1f}; {2:.1f}; {3:.1f}; {4:.1f} <-> {5:.1f}; {6:.1f}; {7:.1f}; {8:.1f} <-> {9};{10};{11};{12}".format(
            "{0} (S_DA = {1:.1f})".format(s.id, s.drainage_km2),
            float(acc_area[indices[0]]),
            float(acc_area[indices[1]]),
            float(acc_area[indices[2]]),
            float(acc_area[indices[3]]),
            float( dists[0] ),
            float(dists[1]),
            float(dists[2]),
            float(dists[3]),
            "({0}, {1})".format(i_indices[indices[0]] + 1, j_indices[indices[0]] + 1),
            "({0}, {1})".format(i_indices[indices[1]] + 1, j_indices[indices[1]] + 1),
            "({0}, {1})".format(i_indices[indices[2]] + 1, j_indices[indices[2]] + 1),
            "({0}, {1})".format(i_indices[indices[3]] + 1, j_indices[indices[3]] + 1)
        ))

    basemap.drawcoastlines(linewidth=0.5)



    xmin, xmax = min(sx), max(sx)
    ymin, ymax = min(sy), max(sy)

    marginx = (xmax - xmin) * 0.1
    marginy = (ymax - ymin) * 0.1
    xmin -= marginx * 1.5
    xmax += marginx * 2
    ymin -= marginy
    ymax += marginy * 2

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    basin_boundaries.plot_basin_boundaries_from_shape(basemap=basemap, plotter=plt, linewidth=1)
    plt.savefig("10yr_cont_stations_natural_fs0.5.pdf")
    #plt.show()


    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(width_pt=None, width_cm=16, height_cm=21)
    main()
    print "Hello world"
  