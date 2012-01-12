__author__ = 'huziy'

import numpy as np
import Ngl

from plot2D.map_parameters import polar_stereographic
import application_properties
from data import data_select
DEFAULT_PATH =  "data/streamflows/hydrosheds_rk4_changed_partiotioning/aex_discharge_1970_01_01_00_00.nc"
def main(data_path = DEFAULT_PATH):
    #get data to memory
    [data, times, x_indices, y_indices] = data_select.get_data_from_file(data_path)
    the_mean = np.mean(data, axis = 0)

    lons2d, lats2d = polar_stereographic.lons, polar_stereographic.lats
    lons = lons2d[x_indices, y_indices]
    lats = lats2d[x_indices, y_indices]


    #colorbar
    wres = Ngl.Resources()
    wres.wkColorMap = "BlGrYeOrReVi200"

    wks_type = "ps"
    wks = Ngl.open_wks(wks_type,"test_pyngl", wres)


    #plot resources
    res = Ngl.Resources()
    res.cnFillMode          = "RasterFill"
    #res.cnFillOn               = True          # Turn on contour fill
    #res.cnMonoFillPattern     = True     # Turn solid fill back on.
    #res.cnMonoFillColor       = False    # Use multiple colors.
    res.cnLineLabelsOn        = False    # Turn off line labels.
    res.cnInfoLabelOn         = False    # Turn off informational
    res.pmLabelBarDisplayMode = "Always" # Turn on label bar.
    res.cnLinesOn             = False    # Turn off contour lines.


    res.mpProjection = "LambertConformal"
    res.mpDataBaseVersion = "MediumRes"


#    res.mpLimitMode         = "LatLon"     # limit map via lat/lon
#    res.mpMinLatF           =  np.min(lats)         # map area
#    res.mpMaxLatF           =  np.max(lats)         # latitudes
#    res.mpMinLonF           =  np.min( lons )         # and
#    res.mpMaxLonF           =  np.max( lons )         # longitudes





    print np.min(lons), np.max(lons)



    res.tiMainFont      = 26
    res.tiXAxisFont     = 26
    res.tiYAxisFont     = 26

    res.sfXArray = lons2d
    res.sfYArray = lats2d
    #
    # Set title resources.
    #
    res.tiMainString         = "Logarithm of mean annual streamflow m**3/s"

    to_plot = np.ma.masked_all(lons2d.shape)
    to_plot[x_indices, y_indices] = np.log(the_mean[:])
#    for i, j, v in zip(x_indices, y_indices, the_mean):
#        to_plot[i, j] = v
    Ngl.contour_map(wks, to_plot[:,:], res)
    Ngl.end()



    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  