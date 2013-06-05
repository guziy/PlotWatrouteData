from multiprocessing.process import Process
import os
import itertools
from scipy.spatial.kdtree import KDTree
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from netCDF4 import MFDataset, Dataset, date2num, Variable
from netCDF4 import num2date

from plot2D.map_parameters import polar_stereographic

def interpolate_to_amno(data_folder, start_year = 1970, end_year = 1999, rcm = "", gcm = "", out_folder = ""):
    print "data_folder: {0}".format( data_folder )
    srof_pattern = os.path.join(data_folder, "mrros_*_*_*.nc")
    trof_pattern = os.path.join(data_folder, "mrro_*_*_*.nc")
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

    time_indices = np.where(
      np.array( map(lambda x: start_year <= x.year <= end_year, times), dtype=np.bool )
    )[0]

    srof_sub = srof_ds.variables["mrros"][time_indices,:,:]
    trof_sub = trof_ds.variables["mrro"][time_indices,:,:]

    times_sub = itertools.ifilter(lambda x: start_year <= x.year <= end_year, times)
    print("selected time window data")


    #writing result to netcdf
    sim_folder = os.path.join(out_folder, "{0}-{1}_{2}-{3}".format(gcm, rcm, start_year, end_year))
    #create a folder for each simulation
    if not os.path.isdir(sim_folder):
        os.mkdir(sim_folder)

    out_path = os.path.join(sim_folder, "narccap_runoff_{0}-{1}_{2}-{3}.nc".format(start_year, end_year, gcm, rcm))
    out_nc = Dataset(out_path, "w")

    out_nc.createDimension("x", polar_stereographic.lons.shape[0])
    out_nc.createDimension("y", polar_stereographic.lats.shape[1])
    out_nc.createDimension("time")

    srof_var = out_nc.createVariable("mrros", "f4", dimensions=("time", "x", "y"))
    trof_var = out_nc.createVariable("mrro", "f4", dimensions=("time", "x", "y"))


    assert isinstance(srof_var, Variable)

    srof_in_var = srof_ds.variables["mrros"]
    for attr_name in srof_in_var.ncattrs():
        print attr_name
        srof_var.setncattr(attr_name, getattr(srof_in_var, attr_name))

    trof_in_var = trof_ds.variables["mrro"]
    for attr_name in trof_in_var.ncattrs():
        print attr_name
        trof_var.setncattr(attr_name, getattr( trof_in_var, attr_name))


    t_var = out_nc.createVariable("time", "f4", dimensions=("time",))
    lon_var = out_nc.createVariable("longitude", "f4", dimensions=( "x", "y"))
    lat_var = out_nc.createVariable("latitude", "f4", dimensions=("x", "y"))


    t_var.units = time_var.units
    print("interpolating and saving data to netcdf file")
    nrows, ncols = polar_stereographic.lons.shape

    #interpolate in time if necessary
    n_interps = 0
    for i, t in enumerate( times_sub ):
        sr_slice = srof_sub[i,:,:].flatten()
        tr_slice = trof_sub[i, :, :].flatten()

        trof1 = tr_slice[indices].reshape(nrows, ncols)
        srof1 = sr_slice[indices].reshape(nrows, ncols)


        if hasattr(trof1, "mask") and np.all(trof1.mask):
            trof1 = trof_var[i-1,:,:]
            n_interps += 1
        if hasattr(srof1, "mask") and np.all(srof1.mask):
            srof1 = srof_var[i-1,:,:]

        trof_var[i,:,:] = trof1
        srof_var[i,:,:] = srof1
        t_var[i] = date2num(t, time_var.units)

    print "Number of interpolations in time: {0}".format(n_interps)
    lon_var[:] = polar_stereographic.lons
    lat_var[:] = polar_stereographic.lats
    out_nc.close()



def main(gcm = "", rcm = "", out_folder = ""):
    kwargs = {
        "start_year": 1970,
        "end_year": 1999,
        "rcm" : rcm, "gcm" : gcm,
        "out_folder" : out_folder
    }
    in_folder = "data/narccap/{0}-{1}/current".format(gcm, rcm)

    if os.path.isdir(in_folder):
        pc = Process(target=interpolate_to_amno, args=(in_folder, ), kwargs = kwargs )
        pc.start()
    else:
        print "{0} does not exist, ignoring the period ...".format(in_folder)

    kwargs = {
        "start_year": 2041,
        "end_year": 2070,
        "rcm" : rcm, "gcm" : gcm,
        "out_folder" : out_folder
    }
    in_folder =  "data/narccap/{0}-{1}/future".format(gcm, rcm)
    if os.path.isdir(in_folder):
        pf = Process(target=interpolate_to_amno, args=(in_folder, ), kwargs = kwargs )
        pf.start()
    else:
        print "{0} does not exist, ignoring the period ...".format(in_folder)

    #do current and future climates in parallel



    #pc.join()
    #pf.join()
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()

    import time
    t0 = time.clock()



    gcm_list = ["ccsm",  "ccsm",  "cgcm3",  "cgcm3",  "cgcm3",  "gfdl",  "gfdl",  "gfdl",  "hadcm3"]
    rcm_list = ["crcm",  "wrfg",  "crcm",  "rcm3",  "wrfg",  "ecp2",  "hrm3",  "rcm3",  "hrm3"]

    for gcm, rcm in zip(gcm_list, rcm_list):
        main(gcm = gcm, rcm = rcm, out_folder = "/home/huziy/skynet1_rech3/narccap_prepared_runoff_for_routing")
    print("Execution time {0} seconds".format(time.clock() - t0))
    print "Hello world"
  
