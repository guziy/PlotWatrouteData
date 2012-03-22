from multiprocessing.process import Process
import os
import itertools
from scipy.spatial.kdtree import KDTree
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from netCDF4 import MFDataset, Dataset, date2num
from netCDF4 import num2date

from plot2D.map_parameters import polar_stereographic

def interpolate_to_amno(data_folder, start_year = 1970, end_year = 1999):
    srof_pattern = os.path.join(data_folder, "mrros_WRFG_ccsm_*.nc")
    trof_pattern = os.path.join(data_folder, "mrro_WRFG_ccsm_*.nc")
    srof_ds = MFDataset(srof_pattern)
    trof_ds = MFDataset(trof_pattern)


    lon_in = srof_ds.variables["lon"][:]
    lat_in = srof_ds.variables["lat"][:]


    x_in, y_in, z_in = lat_lon.lon_lat_to_cartesian(lon_in.flatten(), lat_in.flatten())

    tree = KDTree(zip(x_in, y_in, z_in))


    lon_out, lat_out = polar_stereographic.lons.flatten(), polar_stereographic.lats.flatten()
    x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lon_out, lat_out)
    distances, indices = tree.query(zip(x_out, y_out, z_out))


    time_var = srof_ds.variables["time"]
    time_in_units = time_var[:]

    times = num2date(time_in_units, time_var.units)

    time_mask = np.array( map(lambda x: start_year <= x.year <= end_year, times) )

    srof_sub = srof_ds.variables["mrros"][time_mask,:,:]
    trof_sub = trof_ds.variables["mrro"][time_mask,:,:]

    times_sub = itertools.ifilter(lambda x: start_year <= x.year <= end_year, times)
    print("selected time window data")


    #writing result to netcdf
    out_nc = Dataset("narccap_runoff_{0}_{1}.nc".format(start_year, end_year), "w")

    out_nc.createDimension("x", polar_stereographic.lons.shape[0])
    out_nc.createDimension("y", polar_stereographic.lats.shape[1])
    out_nc.createDimension("time")

    srof_var = out_nc.createVariable("mrros", "f4", dimensions=("time", "x", "y"))
    trof_var = out_nc.createVariable("mrro", "f4", dimensions=("time", "x", "y"))
    t_var = out_nc.createVariable("time", "f4", dimensions=("time",))
    lon_var = out_nc.createVariable("longitude", "f4", dimensions=( "x", "y"))
    lat_var = out_nc.createVariable("latitude", "f4", dimensions=("x", "y"))


    t_var.units = time_var.units
    print("interpolating and saving data to netcdf file")
    nrows, ncols = polar_stereographic.lons.shape
    for i, t in enumerate( times_sub ):
        sr_slice = srof_sub[i,:,:].flatten()
        tr_slice = trof_sub[i, :, :].flatten()
        trof_var[i,:,:] = tr_slice[indices].reshape(nrows, ncols)
        srof_var[i,:,:] = sr_slice[indices].reshape(nrows, ncols)
        t_var[i] = date2num(t, time_var.units)

    lon_var[:] = polar_stereographic.lons
    lat_var[:] = polar_stereographic.lats
    out_nc.close()



def main():
    kwargs = {
        "start_year": 1970,
        "end_year": 1999
    }
    pc = Process(target=interpolate_to_amno, args=("data/narccap/ccsm-wrfg/current",), kwargs = kwargs )

    kwargs = {
        "start_year": 2041,
        "end_year": 2070
    }
    pf = Process(target=interpolate_to_amno, args=("data/narccap/ccsm-wrfg/future",), kwargs = kwargs )

    #do current and future climates in parallel
    pc.start()
    pf.start()

    pc.join()
    pf.join()
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()

    import time
    t0 = time.clock()
    main()
    print("Execution time {0} seconds".format(time.clock() - t0))
    print "Hello world"
  