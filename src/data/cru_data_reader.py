from data import data_select
from util import plot_utils

__author__ = 'huziy'


import netCDF4 as nc
from datetime import datetime
from datetime import timedelta
import application_properties
import numpy as np
from scipy.spatial import KDTree
from util.geo import lat_lon

import pandas

class CruReader():
    def __init__(self, path ="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc",
                 var_name = "pre", create_tree_for_interpolation = True, transpose_xy_dimensions = True):
        """
        Class for reading cru data
        """
        self.path = path
        self.transpose_xy_dimensions = transpose_xy_dimensions
        self.ds = nc.Dataset(path)
        self._read_times()
        self._read_coords()
        self._read_data(var_name = var_name)
        if create_tree_for_interpolation:
            self._set_kd_tree()

    def get_time_step(self):
        """
        :rtype: datetime.timedelta
        """
        return self.times[1] - self.times[0]

    def _read_times(self):
        t_days = self.ds.variables["time"]
        #ref_date = datetime.strptime(t_days.units.split()[2], "%Y-%m-%d")
        self.times = nc.num2date(t_days[:], t_days.units)
        #map(lambda dt : ref_date + timedelta(days = dt), t_days[:])
        print "cru, number of datetime objects: ",len(self.times)


    def get_monthly_normals(self, start_date = None, end_date = None):
        """
        :type start_date: datetime.datetime
        :type end_date: datetime.datetime

        returns the array with shape (month, lon, lat)
        """
        start_date = self.times[0] if start_date is None else start_date
        end_date = self.times[-1] if end_date is None else end_date

        result = np.zeros((12,) + self.data.shape[1:])

        for month in xrange(1,13):
            indices = map(
               lambda x: start_date <= x <= end_date and
                         x.month == month, self.times
            )
            indices = np.where(indices)[0]
            result[month-1] = np.mean(self.data[indices, :, :], axis=0)

        return result


    def get_spatial_integral_for_points(self, the_lons, the_lats):
        """
        get the sum over points, and return timeseries
        """
        x = self.get_time_series_for_points(the_lons, the_lats)
        return np.sum(x, axis = 1)



    def get_daily_normals_integrated_over(self, the_lons, the_lats):
        """
        returns 2 lists
        1) dates of the stamp year
        2) values corresponding to the dates
        """
        data = self.get_spatial_integral_for_points(the_lons, the_lats)

        normals = {}
        year = 2001
        for t, x in zip(self.times, data):
            #skip the 29th of february
            if t.month == 2 and t.day == 29: continue
            d = datetime(year, t.month, t.day, 0, 0, 0)
            if normals.has_key(d):
                normals[d] += [x]
            else:
                normals[d] = [x]

        the_dates = normals.keys()
        the_dates.sort()
        self.stamp_dates = the_dates
        return [np.mean(normals[d]) for d in the_dates]
        pass

    def get_monthly_normals_integrated_over(self, the_lons, the_lats, start_date = None, end_date = None):
        data = self.get_spatial_integral_for_points(the_lons, the_lats)
        monthly_normals = np.zeros((12,))
        for t, x in zip(self.times, data):
            if start_date is not None:
                if t < start_date:
                    continue
            if end_date is not None:
                if t > end_date:
                    break

            monthly_normals[t.month - 1] += x

        start = start_date if start_date is not None else self.times[0]
        end = end_date if end_date is not None else self.times[-1]
        dt = end - start
        n_years = float( dt.days // 365 )
        monthly_normals /= n_years
        return monthly_normals


        
    def get_time_series_for_points(self, the_lons, the_lats):
        """
        get data, data is a 2d array of data with the axes (time, point),
        the point is in the same order as the_lons and the_lats passed to the method
        """


        result = []
        for lon, lat in zip(the_lons, the_lats):
            dlons = np.abs(self.lons - lon)
            dlats = np.abs(self.lats - lat)

            i = np.where(dlons == np.min(dlons))
            j = np.where(dlats == np.min(dlats))

            print i, j

            result.append(self.data[:, i[0][0], j[0][0]])

        result = np.array(result)
        print result.shape
        result = np.transpose(result)
        return result


    def close_data_connection(self):
        """
        Closes 
        """
        self.ds.close()

    def _read_coords(self):
        """
        read longitudes and latitudes from file
        """
        lon_name = "lon"
        lat_name = "lat"
        var_names = self.ds.variables.keys()
        if lon_name not in var_names:
            lon_name = "longitude"

        if lat_name not in var_names:
            lat_name = "latitude"

        self.lons = self.ds.variables[lon_name][:]
        self.lats = self.ds.variables[lat_name][:]
        if len(self.lons.shape) == 1:
            self.lats_2d, self.lons_2d = np.meshgrid(self.lats, self.lons)
        else:
            self.lats_2d, self.lons_2d = self.lats, self.lons

    def _read_data(self, var_name = "pre"):
        """
        gets data from the file
        :type var_name : str

        """
        #try all uppercase just in case
        if not self.ds.variables.has_key(var_name):
            var_name = var_name.upper()
        precip = self.ds.variables[var_name]
        self.data = precip[:]
        #I prefer longitude x-axis
        if self.transpose_xy_dimensions:
            self.data = np.transpose( self.data, axes = (0, 2, 1))

        if hasattr(precip, "missing_value"):
            self.missing_data = precip.missing_value

    def print_file_info(self):
        if hasattr(self, "times"):
            print "start date: ", self.times[0]
            print "end date: ", self.times[-1]
        if len(self.lons.shape) == 1:
            dlon = self.lons[1] - self.lons[0]
            dlat = self.lats[1] - self.lats[0]
            print "dlon, dlat = {0},{1}".format( dlon, dlat )




    def _set_kd_tree(self):

        if len(self.lons.shape) == 1:
            lats_2d, lons_2d = np.meshgrid(self.lats, self.lons)
        else:
            lats_2d, lons_2d = self.lats, self.lons

        x, y, z = lat_lon.lon_lat_to_cartesian(lons_2d.flatten(), lats_2d.flatten())
        print lons_2d.shape, lats_2d.shape
        print lons_2d[0,0], lons_2d[-1, 0]
        print self.data.shape
        self.kdtree = KDTree(data = zip(x, y, z))

    def interpolate_data_to(self, data_in, lons2d, lats2d, nneighbors = 4):
        """
        Interpolates data_in to the grid defined by (lons2d, lats2d)
        assuming that the data_in field is on the initial CRU grid

        interpolate using 4 nearest neighbors and inverse of squared distance
        """

        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())

        dst, ind = self.kdtree.query(zip(x_out, y_out, z_out), k=nneighbors)

        data_in_flat = data_in.flatten()

        inverse_square = 1.0 / dst ** 2
        if len(dst.shape) > 1:
            norm = np.sum(inverse_square, axis=1)
            norm = np.array( [norm] * dst.shape[1] ).transpose()
            coefs = inverse_square / norm
            data_out_flat = np.sum( coefs * data_in_flat[ind], axis= 1)
        elif len(dst.shape) == 1:
            data_out_flat = data_in_flat[ind]
        else:
            raise Exception("Could not find neighbor points")
        return np.reshape(data_out_flat, lons2d.shape)

    def get_seasonal_mean_field(self, months = None, start_date = None, end_date = None):
        if start_date is None: start_date = self.times[0]
        if end_date is None: end_date = self.times[-1]
        bool_vector = np.where(map( lambda x: (x.month in months) and
                                              (start_date <= x) and
                                              (x <= end_date), self.times))[0]
        return np.mean(self.data[bool_vector, :, :], axis=0)


    def get_temporal_evol_of_mean_over(self, dest_lons = None, dest_lats = None, start_date = None, end_date = None):
        """
        interpolates to (dest_lons, dest_lats) and then takes mean over domain for
        the times between start_date and end_date.

        """

        pass


def _get_routing_indices():
    """
    Used for the plot domain centering
    """
    i_indices, j_indices = data_select.get_indices_from_file(path = "data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc")
    return i_indices, j_indices


def test():
    cr = CruReader(path= "data/swe_ross_brown/swe.nc", transpose_xy_dimensions=False, var_name="SWE")

    cr.print_file_info()
    import matplotlib.pyplot as plt
    from plot2D.map_parameters import polar_stereographic

    djf =  cr.get_seasonal_mean_field(months=[1,2,12], start_date=datetime(1980,1,1), end_date=datetime(1997,12,31))
    djf = cr.interpolate_data_to(djf, polar_stereographic.lons, polar_stereographic.lats, nneighbors=1)
    x, y = polar_stereographic.xs, polar_stereographic.ys
    i_array, j_array = _get_routing_indices()
    x_min, x_max, y_min, y_max = plot_utils.get_ranges(x[i_array, j_array], y[i_array, j_array])

    save = djf[i_array, j_array]
    djf = np.ma.masked_all(djf.shape)
    djf[i_array, j_array] = save
    plt.pcolormesh(x, y, djf)
    plt.colorbar()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
 #   cr.interpolate_to()

if __name__ == '__main__':
    application_properties.set_current_directory()
    test()

  