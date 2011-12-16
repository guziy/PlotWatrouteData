
__author__ = 'huziy'


import netCDF4 as nc
from datetime import datetime
from datetime import timedelta
import application_properties
import numpy as np
from scipy.spatial import KDTree
from util.geo import lat_lon

class CruReader():
    def __init__(self, path ="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc"):
        """
        Class for reading cru data
        """
        self.path = path
        self.ds = nc.Dataset(path)
        self._read_times()
        self._read_coords()
        self._read_data()
        self._set_kd_tree()


    def _read_times(self):
        t_days = self.ds.variables["time"]
        ref_date = datetime.strptime(t_days.units.split()[2], "%Y-%m-%d")
        self.times = map(lambda dt : ref_date + timedelta(days = dt), t_days[:])
        print "cru, number of datetime objects: ",len(self.times)
        print( ref_date )


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
            d = self.data[:, i[0][0], j[0][0]] - self.missing_data
            assert not np.any(d == 0)

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
        self.lons = self.ds.variables["lon"][:]
        self.lats = self.ds.variables["lat"][:]

    def _read_data(self):
        """
        gets data from the file

        """
        precip = self.ds.variables["pre"]
        self.data = precip[:]
        #I prefer longitude x-axis
        self.data = np.transpose( self.data, axes = (0, 2, 1))
        self.missing_data = precip.missing_value

    def print_file_info(self):
        if hasattr(self, "times"):
            print "start date: ", self.times[0]
            print "end date: ", self.times[-1]
        dlon = self.lons[1] - self.lons[0]
        dlat = self.lats[1] - self.lats[0]
        print "dlon, dlat = {0},{1}".format( dlon, dlat )




    def _set_kd_tree(self):
        lats_2d, lons_2d = np.meshgrid(self.lats, self.lons)
        print lons_2d.shape, lats_2d.shape
        print lons_2d[0,0], lons_2d[-1, 0]
        print self.data.shape
        self.kdtree = KDTree(data = zip(lons_2d.ravel(), lats_2d.ravel()))

    def interpolate_to(self, dest_lons = None, dest_lats = None):
        """
        Interpolates using IDW (Shepards formula)
        method from (self.lons, self.lats) to (dest_lons, dest_lats)

        """
        distances, indices = self.kdtree.query([10.0,10.0], k=4)


        #Determine neighbor indices and weights for each destination point
        neighbor_weights = [] #n_dest_points x 4 (4 weights for each destination point)
        neighbor_indices = [] #of the same shape as neighbor_weights
        grid_lon_lat = self.data
        for dest_lon, dest_lat in zip(dest_lons, dest_lats):
            deg_dists, indices = self.kdtree.query([dest_lon, dest_lat], k=4)

            neighbor_distances = []
            for the_index in  indices:
                lon, lat = grid_lon_lat[the_index]
                d = lat_lon.get_distance_in_meters(lon, lat, dest_lon, dest_lat)
                neighbor_distances.append(d)

            neighbor_distances = np.array(neighbor_distances)


            neighbor_distances.append(neighbor_distances)
            neighbor_indices.append(indices)


        if not hasattr(indices,  "__iter__"):
           indices = [indices]
        for theIndex in indices:
            print self.kdtree.data[theIndex]

        #for all times, do the interpolation
        for t in xrange(self.data.shape[0]):
            pass


        print self.kdtree.data[-1]
        print self.kdtree.data[0]
        print len(self.kdtree.data)


    def get_temporal_evol_of_mean_over(self, dest_lons = None, dest_lats = None, start_date = None, end_date = None):
        """
        interpolates to (dest_lons, dest_lats) and then takes mean over domain for
        the times between start_date and end_date.

        """

        pass


def test():
    cr = CruReader()
    cr.print_file_info()
 #   cr.interpolate_to()

if __name__ == '__main__':
    application_properties.set_current_directory()
    test()

  