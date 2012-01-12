import os
from osgeo import ogr
import application_properties
from ecmwf.ecmwf_netcdf_reader import EcmwfReader
from util import plot_utils

__author__ = 'huziy'


from ccc.ccc import champ_ccc
import numpy as np
from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
from data import quebec_route_domain_crcm4
from plot2D.map_parameters import polar_stereographic

def _get_month_from_filename(file_name):
    s = file_name.split("_")[-1].split(".")[0]
    return datetime.strptime(s, "%Y%m").month

def calculate_mean_evap_integrated_over(mask = None, data_folder = "data/ccc_data/aex/aex_p1qfs"):

    dt = timedelta( hours = 6 )  #data time step

    data_for_months = [[] for i in xrange(12) ]
    fields_for_months = [[] for i in xrange(12) ]

    for the_file in os.listdir(data_folder):
        ccc_obj = champ_ccc( fichier = os.path.join(data_folder, the_file) )
        records = ccc_obj.charge_champs()

        the_month = _get_month_from_filename(the_file)
        #each record is a step in time
        domain_mean_series = []
        data_fields = []
        for the_record in records:
            data_fields.append( the_record["field"] )
            datai = np.mean(the_record["field"][mask == 1]) #get mean for domain
            domain_mean_series.append(datai)

        data_for_months[the_month - 1].append(np.sum(domain_mean_series) * dt.seconds / 1000.0)
        fields_for_months[the_month - 1].append( np.sum(data_fields, axis = 0) * dt.seconds / 1000.0 )

    for i in xrange(12):
        data_for_months[i] = np.mean(data_for_months[i])
        fields_for_months[i] = np.mean(fields_for_months[i], axis = 0)

    print np.sum(data_for_months)

    plt.figure()
    plt.plot(xrange(1,13), data_for_months)
    plt.savefig("evap_qc_crcm4.png")


    plt.figure()
    b = polar_stereographic.basemap
    [x, y] = b(polar_stereographic.lons, polar_stereographic.lats)
    mean_annual_field = np.sum(fields_for_months, axis = 0)
    b.pcolormesh(x,y,np.ma.masked_where(mask != 1, mean_annual_field))
    print np.mean(mean_annual_field[mask == 1])
    b.drawcoastlines()
    plt.colorbar()
    r = plot_utils.get_ranges(x[mask == 1], y[mask == 1])
    plt.xlim(r[:2])
    plt.ylim(r[2:])
    plt.savefig("evap_qc_2d_crcm4.png")

if __name__ == "__main__":
    application_properties.set_current_directory()
    #calculate_mean_evap_integrated_over(mask=quebec_route_domain_crcm4.get_domain_mask())
    er = EcmwfReader()
    mask = np.zeros(polar_stereographic.lons.shape)
    feature = er.get_qc_polygon_object()
    qc_polygon = feature.GetGeometryRef()

    lons = polar_stereographic.lons
    lats = polar_stereographic.lats
    [nx, ny] = lons.shape
    for i in xrange(nx):
        for j in xrange(ny):
            lon = lons[i,j]
            lat = lats[i,j]
            p = ogr.CreateGeometryFromWkt('POINT('+str(lon) + ' ' + str(lat) + ')')
            mask[i, j] = int( qc_polygon.Contains(p) )

    print "Calculating mean evaporation"
    calculate_mean_evap_integrated_over(mask=mask)