__author__ = 'huziy'


import netCDF4 as nc
from datetime import datetime
from datetime import timedelta
import application_properties
import numpy as np


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

    def get_monthly_normals_integrated_over(self, the_lons, the_lats):
        data = self.get_spatial_integral_for_points(the_lons, the_lats)
        monthly_normals = np.zeros((12,))
        for t, x in zip(self.times, data):
            monthly_normals[t.month - 1] += x

        dt = self.times[-1] - self.times[0]
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


def test():
    cr = CruReader()


if __name__ == '__main__':
    application_properties.set_current_directory()
    test()

  