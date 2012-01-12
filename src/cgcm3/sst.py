import itertools
from mpl_toolkits.basemap import Basemap
from data import data_select

import matplotlib.pyplot as plt
from util import plot_utils

__author__ = 'huziy'


import os
import application_properties
from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta
import numpy as np

from scipy.spatial import KDTree

current_to_start_year = {
    True : 1970,
    False : 1999
}

current_to_end_year = {
    True : 2041,
    False : 2070
}



class DataHolder:
    def __init__(self, data_folder_path = "data/cgcm3",
                 var_name = "tos",
                 ):

        self.data_folder_path = data_folder_path
        self.current_times = None
        self.future_times = None
        self.current_data = None
        self.lons = None
        self.lats = None
        self.var_name = var_name
        self._attach_input_files()
        self._read_lat_lon_and_times()


        self.basemap = Basemap(projection="ortho", lon_0=320, lat_0=50)

        #for interpolation
        #self.kdtree = KDTree(  )

        pass


    def get_data_interpolated_on_amno(self, lons = None, lats = None):
        """
        returns data interpolated to the givel lons and lats
        uses closest neighbour interpolation
        """
        #TODO: needed if you want to route using the runoff from global models

        pass



    def _find_input_files(self, containing_folder):
        """
        find the paths to files containing the variable self.var_name
        returns list of paths
        """
        paths = []
        for dir_name in os.listdir(containing_folder):
            dir_path = os.path.join(containing_folder, dir_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.startswith(self.var_name):
                        file_path =  os.path.join(dir_path, file_name)
                        paths.append(file_path)
                        print file_path
                        break #no more than 1 file per directory
        return paths


    def _attach_input_files(self):
        future_folder = os.path.join(self.data_folder_path, "future")
        self.future_files = self._find_input_files(future_folder)

        current_folder = os.path.join(self.data_folder_path, "current")
        self.current_files = self._find_input_files(current_folder)
        pass


    def _netcdf_var_to_datetime_objs(self, the_var):
        units_str = the_var.units.strip()

        dt_s, start_date_s = units_str.split("since")

        data = the_var[:]
        if dt_s.startswith("days"):
            dt = timedelta( hours = 1 )
            data = map(int, data * 24)
        elif dt_s.startswith("hours"):
            dt = timedelta( hours = 1 )
        else:
            raise Exception("Unknown units string")

        date_format = "%Y-%M-%d"
        start_date = datetime.strptime(start_date_s.strip(), date_format)


        return map( lambda n: start_date + n * dt, data)


    def _read_lat_lon_and_times(self):
        the_path = self.current_files[0]
        ds = Dataset(the_path)
        lon1d = ds.variables["lon"][:]

        #lon1d[lon1d > 180] -= 360

        lat1d = ds.variables["lat"][:]


        self.lats, self.lons = np.meshgrid(lat1d, lon1d)
        print (lon1d.shape, lat1d.shape)
        print self.lons.shape, self.lats.shape

        time_var = ds.variables["time"]
        self.current_times = self._netcdf_var_to_datetime_objs(time_var)

        ds = Dataset( self.future_files[0] )
        time_var = ds.variables["time"]
        self.future_times = self._netcdf_var_to_datetime_objs(time_var)




    def get_cv_for_seasonal_mean(self, months = range(1, 13),
                                 current = True,
                                 start_date = None,
                                 end_date = None
                                 ):
        """
        returns cv calculated for the period of year defined by months
        """
        paths = self.current_files if current else self.future_files
        times = self.current_times if current else self.future_times

        time_start_index = 0
        if start_date is not None:
            times_select = itertools.ifilter(lambda x: x >= start_date, times)
            times_select = list(times_select)
            time_start_index = times.index(times_select[0])

        time_end_index = len(times) - 1
        if end_date is not None:
            times_select = itertools.ifilter(lambda x: x <= end_date, times)
            times_select = list( times_select )
            time_end_index = times.index( times_select[-1] )

        times = times[time_start_index:time_end_index + 1]

        all_data = []
        for the_path in paths:
            ds = Dataset(the_path)


            the_var = ds.variables[self.var_name]
            the_data = the_var[:]
            #transpose in order to have (time, lon, lat)
            the_data = np.transpose(the_data, axes = (0,2,1))
            #subset along time dimension
            the_data = the_data[time_start_index:time_end_index + 1,:,:]

            bool_vector = map(lambda x: x.month in months, times)
            indices = np.where(bool_vector)[0]

            if hasattr(the_var, "missing_value"):
                the_data = np.ma.masked_where(the_data == the_var.missing_value, the_data)
            all_data.append(np.ma.mean(the_data[indices,:,:], axis=0))


        #calculate cv
        all_data = np.ma.array(all_data)

        stds = np.ma.std(all_data, axis = 0)
        means = np.ma.mean(all_data, axis = 0)
        return np.ma.divide( stds, means )


    def plot_2d_field(self, field_2d = None, file_name = "", color_levels = np.arange(0,1.0, 0.25)):
        """

        """
        [x, y] = self.basemap( self.lons, self.lats )
        cs = self.basemap.contourf( x, y, field_2d, levels = color_levels)
        self.basemap.drawcoastlines()
        plt.colorbar()
        pass

    def plot_using_ngl(self, field_2d = None, file_name = None):
        import Ngl
        wks_type = "png"
        wks = Ngl.open_wks(wks_type, "cv")
        res = Ngl.Resources()
        res.mpProjection = "Orthographic"

        pass

def main():
    dh = DataHolder(var_name = "tos")
    plot_utils.apply_plot_params(width_pt=None, font_size=9)

    current_start_date = datetime(1970,1,1)
    current_end_date = datetime(1999,12,31)

    future_start_date = datetime( 2041, 1, 1 )
    future_end_date = datetime( 2070, 12, 31)

    plt.subplot(2,1,1)
    plt.title("SST: current CV")

    cv = dh.get_cv_for_seasonal_mean(start_date = current_start_date,
                                     end_date = current_end_date, months=range(3,8))
    dh.plot_2d_field(field_2d= cv,
                        color_levels=np.arange(0, 0.001, 0.0001)
                    )

    plt.subplot(2,1,2)
    plt.title("SST: future CV")
    cv = dh.get_cv_for_seasonal_mean(current = False,
                                        start_date=future_start_date,
                                        end_date = future_end_date, months=range(3,8))
    dh.plot_2d_field(field_2d = cv,
                        color_levels=np.arange(0, 0.001, 0.0001)
                    )




    plt.savefig("sst_cv.png")
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()